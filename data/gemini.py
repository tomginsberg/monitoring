import pickle
from typing import Callable, Any

import numpy as np
import sklearn
import xgboost
import attr
import yaml
from datasets import Dataset
import pandas as pd
from tqdm import tqdm

from globals import *
from utils.dp import train_test_val_split
from time import time
import random

DEFAULT_METADATA = [DISCHARGE_DISPOSITION, LENGTH_OF_STAY, GENC_ID, HOSPITAL_ID, RESIDENCE_CODE, *DATE_TIME_COLS]


@attr.s(auto_attribs=True)
class TrainedModelConfig:
    features: list[str]
    label: str | Callable[[pd.DataFrame, ], np.ndarray]
    test_indices: list[int]
    model: sklearn.base.ClassifierMixin
    hparams: dict[str, Any]

    def dump(self, path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> 'TrainedModelConfig':
        with open(path, 'rb') as f:
            return pickle.load(f)


class Gemini:
    """
        Typical workflow:
        1. Load data out of memory using gem = Gemini(path)
        2. Drop columns using appropriate helper methods
        3. Call gem.load() to get a GeminiDataFrame object (which is a pandas DataFrame)
        that includes several helper methods for data manipulation/analysis
    """

    def __init__(self, path=None, n_bins: int | None = 8,
                 bin_length: int | None = 6):
        if path is None:
            path = DEFAULT_DATA_PATH
        self.path = path
        self.dataset = Dataset.load_from_disk(path).remove_columns('__index_level_0__')
        self.n_bins = n_bins
        self.bin_length = bin_length

    @staticmethod
    def from_config(config: TrainedModelConfig, data_path='/mnt/nfs/home/dshift_project/geminaid/processed_data'):
        gem = Gemini(path=data_path, n_bins=None, bin_length=None)
        gem.select_rows(config.test_indices)
        return gem

    def subsample(self, n=1000, seed=0):
        rng = random.Random(seed)
        indices = rng.sample(range(len(self.dataset)), n)
        self.dataset = self.dataset.select(indices)
        return self

    def select_rows(self, indices):
        self.dataset = self.dataset.select(indices)
        return self

    def filter_rows(self, condition: callable):
        self.dataset = self.dataset.filter(condition)
        return self

    def remove_columns(self, cols):
        self.dataset = self.dataset.remove_columns(cols)
        return self

    def select_columns(self, cols):
        self.dataset = self.dataset.select_columns(cols)
        return self

    def drop_bins(self, start=4, end=7):
        # drop all cols that end with [-{x} for x in range(start, end + 1)]
        cols = [col for x in range(start, end + 1) for col in self.dataset.column_names if col.endswith(f"-{x}")]
        self.remove_columns(cols)
        return self

    def cutoff(self, hours: int = 48):
        """
        Drop all timestamped features (labs, vitals) that occur later than <hours> hours after admission
        :param hours: int, number of hours after admission to cutoff
        :return: self, filtered dataset
        """
        if self.n_bins is not None and self.bin_length is not None:
            assert hours <= (
                max_hrs := self.n_bins * self.bin_length), f'hours must be <= {max_hrs} for the current binning scheme'
        # drop bins that start > hours
        self.drop_bins(start=hours // self.bin_length, end=self.n_bins - 1)
        return self

    def drop_admin_variables(self):
        """
        Drop all administrative variables (genc_id, hospital_id, residence_code, date_time_cols)
        :return: self, filtered dataset
        """
        self.remove_columns([GENC_ID, HOSPITAL_ID, RESIDENCE_CODE, *DATE_TIME_COLS])
        return self

    @property
    def column_names(self):
        return self.dataset.column_names

    @property
    def shape(self):
        return self.dataset.shape

    def drop_labs(self):
        """
        Drop all lab features
        :return: self, filtered dataset
        """
        self.remove_columns([x for x in self.column_names if x.startswith('lab_')])
        return self

    def drop_vitals(self):
        """
        Drop all vital features
        :return: self, filtered dataset
        """
        self.remove_columns([x for x in self.column_names if x.startswith('vital_')])
        return self

    def drop_diagnosis(self):
        """
        Drop all diagnosis features
        :return: self, filtered dataset
        """
        self.remove_columns([x for x in self.column_names if x.startswith('diag_')])
        return self

    def drop_statcan(self):
        """
        Drop all statcan features (income, education, etc.)
        :return: self, filtered dataset
        """
        self.remove_columns(STATCAN_FEATURES)
        return self

    def load(self):
        """
        Load the data into memory as a GeminiDataFrame (extends pd.DataFrame)
        :return: GeminiDataFrame
        """
        return GeminiDataFrame(self)


TASKS = {
    'los-3': lambda x: x[LENGTH_OF_STAY] < 3,
    'los-5': lambda x: x[LENGTH_OF_STAY] < 5,
    'mort-7': lambda x: (x[LENGTH_OF_STAY] < 7) & (x[DISCHARGE_DISPOSITION] == 1),
    'mort-14': lambda x: (x[LENGTH_OF_STAY] < 14) & (x[DISCHARGE_DISPOSITION] == 1),
    'mort-30': lambda x: (x[LENGTH_OF_STAY] < 30) & (x[DISCHARGE_DISPOSITION] == 1),
    'mort-90': lambda x: (x[LENGTH_OF_STAY] < 90) & (x[DISCHARGE_DISPOSITION] == 1),
    'los-median': lambda x: x[LENGTH_OF_STAY] < MEDIAN_LOS
}


@attr.s(auto_attribs=True)
class GroupedDataConfig:
    features: list[str]
    label: str | Callable[[pd.DataFrame, ], np.ndarray]
    grouper: str
    test_indices: dict[str, list[int]]
    test_samples: int

    def to_yaml(self, path):
        with open(path, 'w') as f:
            yaml.dump(attr.asdict(self), f)

    def from_yaml(self, path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.features = config['features']
            self.label = config['label']
            self.grouper = config['grouper']
            self.test_indices = config['test_indices']

    def create_model_config(self, model: xgboost.XGBClassifier, hparams: dict[str, Any], group_name: str):
        # copy this dataconfig but update the test indices as a union of
        # the train/val/test indices for all groups but this one as well as the test indices for this group
        indices = []
        for group, group_indices in self.test_indices.items():
            if group != group_name:
                indices.extend(group_indices)
        indices.extend(self.test_indices[group_name])
        return TrainedModelConfig(
            features=self.features,
            label=self.label,
            test_indices=indices,
            model=model,
            hparams=hparams
        )


class GeminiDataFrame(pd.DataFrame):
    def __init__(self, gem: Gemini):
        super().__init__(gem.dataset.to_pandas())

    def process(self, group_by=HOSPITAL_ID, seed=42, train_test_val=(70, 15, 15),
                task: str | Callable[[pd.DataFrame, ], np.ndarray] = 'los-5',
                drop_cols=DEFAULT_METADATA):
        """

        :param group_by: column to group data by, defaults to HOSPITAL_ID, if None, will group all data together
        :param seed: random seed for the train/test/val split
        :param train_test_val: relative sizes of train/test/val splits
        :param task: mappable function or column to use as the prediction label,
                defaults to 'los-5' which is a binary label for length of stay < 5
                see data.TASKS for the default set of tasks
        :param drop_cols: columns to drop from the data, defaults to data.DEFAULT_METADATA
        :return:
        """
        start = time()
        result = {}

        if task in TASKS:
            task = TASKS[task]

        if group_by is None:
            group_by = 'all'
            self[group_by] = 0

        if group_by in self.keys() and group_by not in drop_cols:
            drop_cols.append(group_by)

        config = GroupedDataConfig(
            features=list(set(self.columns) - set(drop_cols)),
            label=task,
            grouper=group_by,
            test_indices={},
            test_samples=len(self)
        )

        for group_name, group_data in tqdm(self.groupby(group_by)):
            if isinstance(task, str):
                labels = group_data[task].to_numpy()
            # check if it's callable
            elif callable(task):
                labels = task(group_data)
            else:
                raise ValueError(f'Invalid label type {type(task)}')

            group_data = group_data.drop(columns=drop_cols)
            data = group_data.to_numpy()

            split_data, indices = train_test_val_split(data, np.array(labels), sizes=train_test_val, random_state=seed)
            config.test_indices[group_name] = indices['test'].tolist()
            result[group_name] = split_data

        elapsed = time() - start
        print(
            f'Processed {self.shape[0]} x {self.shape[1]} datapoints into {len(result)} groups in {elapsed:.2f} seconds'
        )
        return result, config

    def process_by_time(self, group_by, train_cutoff, val_cutoff,
                        task: str | Callable[[pd.DataFrame, ], np.ndarray] = 'los-5',
                        drop_cols=DEFAULT_METADATA):
        """
        :param group_by: column to group data by
        :param train_cutoff: data before/at train_cutoff will be used as training data
        :param val_cutoff: data after train_cutoff and before/at val_cutoff will be used as validation data
        :param task: mappable function or column to use as the prediction label,
                defaults to 'los-5' which is a binary label for length of stay < 5
                see data.TASKS for the default set of tasks
        :param drop_cols: columns to drop from the data, defaults to data.DEFAULT_METADATA
        :return:
        """
        start = time()
        data_dict = {
            'train': None,
            'val': None,
            'test': None}
        result = {}

        if task in TASKS:
            task = TASKS[task]

        if group_by in self.keys() and group_by not in drop_cols:
            drop_cols.append(group_by)

        config = GroupedDataConfig(
            features=list(set(self.columns) - set(drop_cols)),
            label=task,
            grouper='time',
            test_indices={},
            test_samples=len(self)
        )

        # Split the data into training, validation, and testing sets
        train_data = self[self[group_by] <= train_cutoff]
        val_data = self[(self[group_by] > train_cutoff) & (self[group_by] <= val_cutoff)]
        test_data = self[self[group_by] > val_cutoff]

        for key, data in zip(data_dict.keys(), [train_data, val_data, test_data]):
            # Get the label
            if callable(task):
                labels = task(data)
            else:
                raise ValueError(f'Invalid label type {type(task)}')

            if key == 'test':
                # Group test data by half years
                for period in sorted(data[group_by].unique()):
                    period_data = data[data[group_by] == period]
                    period_data = period_data.drop(columns=drop_cols)
                    period_indices = period_data.index.tolist()
                    data_dict[key] = (period_data.to_numpy(), labels[period_indices].to_numpy())
                    config.test_indices[str(period)] = period_indices
                    result[str(period)] = data_dict.copy()
            else:
                data = data.drop(columns=drop_cols)
                data_dict[key] = (data.to_numpy(), labels.to_numpy())

        elapsed = time() - start
        return result, config


class ConfiguredDataset:
    def __init__(self, config: TrainedModelConfig = None, dataset_path=None,
                 features: np.ndarray = None, labels: np.array = None, metadata: pd.DataFrame = None):
        if features is None and labels is None and metadata is None and config is not None:
            gem = Gemini.from_config(config, dataset_path)
            gem = gem.dataset.to_pandas()
            self.features = gem[config.features].to_numpy()
            self.labels = config.label(gem).to_numpy()
            self.meta_features = list(set(DEFAULT_METADATA + IP_KEYS))
            self.metadata = gem[self.meta_features]
        else:
            self.features = features
            self.labels = labels
            self.metadata = metadata
            self.meta_features = list(metadata.columns)

    def filter_by_metadata(self, query: str):
        """
        Filter the dataset by metadata query
        :param query:  string to filter by, for example 'age > 50'
        :return: a new ConfiguredDataset object with the filtered data
        """
        query_rows = self.metadata.query(query).index
        return ConfiguredDataset(
            features=self.features[query_rows],
            labels=self.labels[query_rows],
            metadata=self.metadata[query_rows],
        )

from datetime import datetime
from time import time
from typing import Any, Union

import pickle
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from utils import generic as generic_utils

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'random_state': 0,
    'eval_metric': 'auc',
    'early_stopping_rounds': 10,
    'n_estimators': 20,
    'max_depth': 7,
}


class Criteria:

    @staticmethod
    def after_date(x, year, month, day):
        return x['date'] >= datetime(year, month, day)

    @staticmethod
    def before_date(x, year, month, day):
        return x['date'] < datetime(year, month, day)

    @staticmethod
    def pre_covid(x):
        return x['date'] < datetime(2020, 3, 1)

    @staticmethod
    def post_covid(x):
        return x['date'] >= datetime(2020, 3, 1)


def get_criteria_set(data: pd.DataFrame, criteria: Union[str,  tuple[str, dict[str, Any]]]) -> pd.DataFrame:
    if isinstance(criteria, tuple):
        criteria, kwargs = criteria
        return Criteria.__dict__[criteria](data, **kwargs)
    else:
        return Criteria.__dict__[criteria](data)


class Tasks:
    pass


def train_val_test_split(n_samples, val_size, test_size, random_state=0):
    train_size = 1 - val_size - test_size
    assert train_size > 0, 'val_size + test_size must be less than 1'
    train_idx, test_idx = train_test_split(
        np.arange(n_samples),
        test_size=test_size,
        random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size / train_size,
        random_state=random_state
    )
    return train_idx, val_idx, test_idx


def process_dataset(
        dataset, input_features,
        label, random_state,
        training_criteria,
        val_size=0.1, test_size=0.4,
        **kwargs
):
    if label in Tasks.__dict__:
        labels = Tasks.__dict__[label](dataset)
    elif label in dataset.columns:
        labels = dataset[label]
        dataset = dataset.drop(columns=[label])
    else:
        raise ValueError(f'Label {label} not found in dataset or in Tasks')
    if isinstance(training_criteria, tuple) or isinstance(training_criteria, list):
        training_criteria, kwargs = training_criteria
        id_data = Criteria.__dict__[training_criteria].__func__(dataset, **kwargs)
    else:
        id_data = Criteria.__dict__[training_criteria](dataset)

    meta_cols = [x for x in dataset.columns if x not in set(input_features)]
    id_idx = np.where(id_data)[0]
    od_idx = np.where(~id_data)[0]
    id_data = dataset[input_features].iloc[id_idx]
    od_data = dataset[input_features].iloc[od_idx]
    id_labels = labels.iloc[id_idx]
    od_labels = labels.iloc[od_idx]
    id_meta = dataset[meta_cols].iloc[id_idx]
    od_meta = dataset[meta_cols].iloc[od_idx]

    train_idx, val_idx, test_idx = train_val_test_split(
        len(id_data),
        val_size,
        test_size,
        random_state=random_state
    )

    return dict(
        train=dict(
            data=id_data.iloc[train_idx],
            label=id_labels.iloc[train_idx],
            meta=id_meta.iloc[train_idx],
        ),
        val=dict(
            data=id_data.iloc[val_idx],
            label=id_labels.iloc[val_idx],
            meta=id_meta.iloc[val_idx],
        ),
        id_test=dict(
            data=id_data.iloc[test_idx],
            label=id_labels.iloc[test_idx],
            meta=id_meta.iloc[test_idx],
        ),
        od_test=dict(
            data=od_data,
            label=od_labels,
            meta=od_meta
        )
    )


class Trainer:

    def __init__(
            self,
            dataset: pd.DataFrame,
            training_criteria: Union[str,  tuple[str, dict[str, Any]]],
            input_features: list[str],
            label: str,
            params: dict = XGB_PARAMS,
            model_class=xgb.XGBClassifier,
            scale_pos_weight=True,
            val_size=0.2,
            test_size=0.3,
            random_state=0,
            verbose=True
    ):
        self.dataset = dataset
        self.training_criteria = training_criteria
        self.input_features = input_features
        self.label = label
        self.hparams = params
        self.verbose = verbose
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.model_class = model_class

        data = process_dataset(
            dataset, input_features=input_features, label=label,
            random_state=random_state, training_criteria=training_criteria,
            val_size=val_size, test_size=test_size
        )
        (self.x_train, self.x_val, self.y_train, self.y_val) = (data['train']['data'], data['val']['data'],
                                                                data['train']['label'], data['val']['label'])
        self.x_test, self.y_test = pd.concat([data['id_test']['data'], data['val']['data']]), pd.concat(
            [data['id_test']['label'], data['val']['label']])

        scale_pos_weight = sum(self.y_train == 0) / sum(self.y_train == 1) if scale_pos_weight else 1
        self.model = model_class(scale_pos_weight=scale_pos_weight, **self.hparams)

    def get_atc_threshold(self):
        predictions = self.model.predict_proba(self.x_test)
        val_acc = accuracy_score(y_true=self.y_test, y_pred=predictions[:, 1] > .5)
        val_auc = roc_auc_score(y_true=self.y_test, y_score=predictions[:, 1])
        val_confidences = np.sort(predictions.max(1))
        thresh = {
            'atc_acc_thresh': val_confidences[int((1 - val_acc) * len(val_confidences)) - 1],
            'atc_auc_thresh': val_confidences[int((1 - val_auc) * len(val_confidences)) - 1]
        }
        print(val_acc, np.mean(val_confidences > thresh['atc_acc_thresh']))
        return thresh

    def fit(self, output_path=None):
        start = time()
        self.model.fit(self.x_train, self.y_train, eval_set=[(self.x_val, self.y_val)], verbose=self.verbose)
        total = time() - start
        config = generic_utils.clean_config(
            dict(
                model=self.model,
                information=dict(
                    hparams=self.hparams,
                    val_size=self.val_size,
                    training_criteria=self.training_criteria,
                    input_features=self.input_features,
                    label=self.label,
                    random_state=self.random_state,
                    scale_pos_weight=self.scale_pos_weight,
                    model_class=self.model_class,
                    datetime_trained=datetime.now(),
                    elapsed_training_time=total,
                    train_samples=len(self.x_train),
                    val_samples=len(self.x_val),
                    test_size=self.test_size,
                    val_auc=roc_auc_score(
                        y_true=self.y_val,
                        y_score=self.model.predict_proba(self.x_val)[:, 1]
                    ),
                    train_auc=roc_auc_score(
                        y_true=self.y_train,
                        y_score=self.model.predict_proba(self.x_train)[:, 1]
                    )

                ) | self.get_atc_threshold()
            )
        )
        if output_path is not None:
            with open(output_path, 'wb') as output_path:
                pickle.dump(config, output_path)

        return config

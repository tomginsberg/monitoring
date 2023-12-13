import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score, accuracy_score


class DetectronDataModule:

    def __init__(self, train_data: np.ndarray | pd.DataFrame,
                 q_data, train_labels, q_pseudo_labels,
                 balance_train_classes=True,
                 test_weight_multiplier=1.0):
        """

        :param train_data:
        :param q_data:
        :param train_labels:
        :param q_pseudo_labels:
        :param balance_train_classes:
        :return:
        """
        self.N = len(q_data)

        self.train_data = train_data
        self.q_data = q_data
        self.train_labels = train_labels
        self.q_pseudo_labels = q_pseudo_labels
        self.alpha = test_weight_multiplier

        if balance_train_classes:
            _, counts = np.unique(train_labels, return_counts=True)
            assert len(counts) == 2, 'Only binary classification is supported in v0.0.1'
            c_neg, c_pos = counts[0], counts[1]
            # make sure the average training weight is 1
            pos_weight, neg_weight = 2 * c_neg / (c_neg + c_pos), 2 * c_pos / (c_neg + c_pos)
            self.train_weights = np.array([pos_weight if label == 1 else neg_weight for label in train_labels])
        else:
            self.train_weights = np.ones_like(train_labels)

    def dataset(self):
        return dict(
            X=np.concatenate([self.train_data, self.q_data]),
            y=np.concatenate([self.train_labels, 1 - self.q_pseudo_labels]),
            sample_weight=np.concatenate(
                [self.train_weights, 1 / (self.N + 1) * np.ones(self.N) * self.alpha]
            )
        )

    def filter(self, detector: xgboost.sklearn.XGBClassifier):
        mask = (detector.predict(self.q_data, validate_features=False) == self.q_pseudo_labels)

        # filter data to exclude the not rejected samples
        self.q_data = self.q_data[mask]
        self.q_pseudo_labels = self.q_pseudo_labels[mask]
        self.N = len(self.q_data)
        return len(self.q_data)


class DetectronRecord:
    def __init__(self, sample_size):
        self.record = []
        self.sample_size = sample_size
        self.idx = 0
        self._seed = None

    def seed(self, seed):
        self._seed = seed
        self.idx = 0

    def update(self, q_labeled, val_data, sample_size, model,
               q_pseudo_probabilities=None):
        assert self._seed is not None, 'Seed must be set before updating the record'
        self.record.append({
            'ensemble_idx': self.idx,
            'val_auc': self.compute_auc(model, val_data),
            'test_auc': self.compute_auc(model, q_labeled),
            'rejection_rate': 1 - sample_size / self.sample_size,
            'test_probabilities': q_pseudo_probabilities if q_pseudo_probabilities is not None else model.predict_proba(
                q_labeled['data'], validate_features=False)[:, 1],
            'count': sample_size,
            'seed': self._seed
        })
        self.idx += 1

    @staticmethod
    def compute_auc(model, data):

        if isinstance(data, dict):
            data = data['data'], data['label']
        try:
            return roc_auc_score(data[1], model.predict_proba(data[0], validate_features=False)[:, 1])
        except ValueError:
            # in case all instances are a single class, compute accuracy instead
            return accuracy_score(data[1], model.predict(data[0], validate_features=False))

    def freeze(self):
        self.record = self.get_record()

    def get_record(self):
        if isinstance(self.record, pd.DataFrame):
            return self.record
        else:
            return pd.DataFrame(self.record)

    def save(self, path):
        self.get_record().to_csv(path, index=False)

    @staticmethod
    def load(path):
        x = DetectronRecord(sample_size=None)
        x.record = pd.read_csv(path)
        x.sample_size = x.record.query('ensemble_idx==0').iloc[0]['count']
        return x

    def counts(self, max_ensemble_size=None) -> np.ndarray:
        assert max_ensemble_size is None or max_ensemble_size > 0, 'max_ensemble_size must be positive or None'
        rec = self.get_record()
        counts = []
        for i in rec.seed.unique():
            run = rec.query(f'seed=={i}')
            if max_ensemble_size is not None:
                run = run.iloc[:max_ensemble_size + 1]
            counts.append(run.iloc[-1]['count'])
        return np.array(counts)

    def count_quantile(self, quantile, max_ensemble_size=None):
        counts = self.counts(max_ensemble_size)
        return np.quantile(counts, quantile, method='inverted_cdf')


def ecdf(x):
    """
    Compute the empirical cumulative distribution function
    :param x: array of 1-D numerical data
    :return: a function that takes a value and returns the probability that
        a random sample from x is less than or equal to that value
    """
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size

    return result


class DetectronResult:
    """
    A class to store the results of a Detectron test
    """

    def __init__(self, cal_record: DetectronRecord, test_record: DetectronRecord):
        """
        :param cal_record: Result of running benchmarking.detectron_test_statistics using IID test data
        :param test_record: Result of running benchmarking.detectron_test_statistics using the unknown test data
        """
        self.cal_record = cal_record
        self.test_record = test_record

        self.cal_counts = cal_record.counts()
        self.test_count = test_record.counts()[0]
        self.baseline = self.cal_counts.mean()
        self.model_health = self.test_count / self.baseline

        self.cdf = ecdf(self.cal_counts)
        self.p_value = self.cdf(self.test_count)

    def calibration_trajectories(self):
        rec = self.cal_record.get_record()
        return rec[['seed', 'ensemble_idx', 'rejection_rate']]

    def test_trajectory(self):
        rec = self.test_record.get_record()
        return rec[['ensemble_idx', 'rejection_rate']]

    def __repr__(self):
        return f"DetectronResult(p_value={self.p_value}, test_statistic={self.test_count}, " \
               f"baseline={self.cal_counts.mean():.2f}±{self.cal_counts.std():.2f})"


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0
        assert mode in ['min', 'max']
        self.mode = mode

    def update(self, metric):
        if self.best is None:
            self.best = metric
            return False
        if self.mode == 'min':
            if metric < self.best - self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience
        elif self.mode == 'max':
            if metric > self.best + self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience

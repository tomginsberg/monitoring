from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from detectron.utils import DetectronDataModule, DetectronRecord, EarlyStopper, DetectronResult
from training.train import XGB_PARAMS


def detectron_test_statistics(
        train: tuple[pd.DataFrame, pd.DataFrame],
        val: tuple[pd.DataFrame, pd.DataFrame],
        q: tuple[pd.DataFrame, pd.DataFrame],
        base_model: xgb.sklearn.XGBClassifier,
        sample_size: int,
        xgb_params=XGB_PARAMS,
        ensemble_size=10,
        calibration_runs=100,
        patience=3,
        test_weight_multiplier=1,
        balance_train_classes=True,
        show_progress_bar=True,
        verbose=True
):
    """
    Run the Detectron algorithm for `seeds` times, and return
    :param train:
    :param val:
    :param q:
    :param base_model:
    :param sample_size:
    :param xgb_params: (trees.defaults.XGB_PARAMS) parameters for the XGBoost model
    :param ensemble_size: (10) the number of models in the ensemble
    :param calibration_runs: (100) the number of different random runs to perform,
        each run operates on a random sample from q
    :param patience: (3) number of ensemble rounds to wait without improvement in the rejection rate
    :param test_weight_multiplier: (1) the weight of the test data in the training set
    :param balance_train_classes: (True) If True, the training data will be automatically balanced using weights.
    :param verbose: (True) If True early stopping and convergence information
    :param show_progress_bar: (True) If True, show a progress bar
    :return: XGBDetectronRecord object containing all the information of this run
    """
    record = DetectronRecord(sample_size)

    # gather the data
    train_data, train_labels = train
    scale_positive_weight = np.mean(train_labels == 0) / np.mean(train_labels == 1)
    val_data, val_labels = val
    val_numpy = [val_data.to_numpy(), val_labels.to_numpy()]
    q_data_all, q_labels_all = q

    for seed in (tqdm if show_progress_bar else lambda x: x)(range(calibration_runs)):

        # randomly sample N elements from q
        idx = np.random.RandomState(seed).permutation(len(q[0]))[:sample_size]
        q_data, q_labels = q_data_all.iloc[idx, :], q_labels_all.iloc[idx]

        # store the test data
        N = len(q_data)
        q_labeled = dict(data=q_data, label=q_labels)

        # evaluate the base model on the test data
        q_pseudo_probabilities = base_model.predict_proba(q_labeled['data'])[:, 1]
        q_pseudo_labels = q_pseudo_probabilities > 0.5

        # create the weighted dataset for training the detectron
        data_module = DetectronDataModule(train_data=train_data,
                                          train_labels=train_labels,
                                          q_data=q_data,
                                          q_pseudo_labels=q_pseudo_labels,
                                          balance_train_classes=balance_train_classes,
                                          test_weight_multiplier=test_weight_multiplier
                                          )

        # evaluate the base model on test and auc data
        record.seed(seed)
        record.update(q_labeled=q_labeled, val_data=val, model=base_model, sample_size=N,
                      q_pseudo_probabilities=q_pseudo_probabilities)

        stopper = EarlyStopper(patience=patience, mode='min')
        stopper.update(N)

        # train the ensemble
        for i in range(1, ensemble_size + 1):
            # train the next model in the ensemble
            xgb_params.update({'seed': i})
            detector = xgb.sklearn.XGBClassifier(scale_pos_weight=scale_positive_weight, **xgb_params)
            detector.fit(
                eval_set=[val_numpy],
                verbose=False,
                **data_module.dataset()
            )

            n = data_module.filter(detector)

            # log the results for this model
            record.update(q_labeled=q_labeled, val_data=val, model=detector, sample_size=n)

            # break if no more data
            if n == 0:
                if verbose:
                    print(f'Converged to a rejection rate of 1 after {i} models')
                break

            if stopper.update(n):
                if verbose:
                    print(f'Early stopping: Converged after {i} models')
                break

    record.freeze()
    return record


def detectron(train: tuple[pd.DataFrame, pd.DataFrame],
              val: tuple[pd.DataFrame, pd.DataFrame],
              observation_data: pd.DataFrame,
              base_model: xgb.sklearn.XGBClassifier,
              calibration_record: DetectronRecord,
              xgb_params=XGB_PARAMS,
              ensemble_size=10,
              patience=3,
              test_weight_multiplier=1,
              balance_train_classes=True,
              show_progress_bar=True,
              verbose=True
              ):
    """
    Perform a Detectron test on a model, using the given data
    :param train: the original split used to train the model
    :param val: the original split used to validate the model
    :param observation_data:
    :param base_model: The base model to use for the pseudo-labeling, this should be a trained XGBoost model
    :param calibration_record: The result of running detectron or benchmarking.detectron_test_statistics on iid_test.
    The calibration data can be collected from a previous detectron run. See the example #1 in
    the README for more details.
    :param xgb_params: (XGB_PARAMS) Parameters to pass to xgboost.train, see
    https://xgboost.readthedocs.io/en/latest/parameter.html
    :param ensemble_size: (10) number of models trained to disagree with each-other.
    Typically, a value of 5-10 is sufficient, but larger values may be required for very large datasets
    :param patience: (3) number of ensemble rounds to wait without an improvement in the rejection rate
    :param balance_train_classes: (True) If True, the training data will be automatically balanced using weights.
    Disable only if your data is already class balanced.
    :param verbose: (True) If True early stopping and convergence information
    :param show_progress_bar: (True) If True, show a progress bar
    :param test_weight_multiplier: (1) the weight of the test data in the training set
    :return: DetectronResult object containing the results of the test.
    """
    sample_size = calibration_record.sample_size
    obs_size = len(observation_data)
    obs_labels = np.zeros(obs_size)

    assert calibration_record.sample_size == obs_size, \
        "The calibration record must have been generated with the same sample size as the observation set"

    test_record = detectron_test_statistics(train=train, val=val, q=(observation_data, pd.DataFrame(obs_labels)),
                                            base_model=base_model, sample_size=sample_size, xgb_params=xgb_params,
                                            ensemble_size=ensemble_size,
                                            calibration_runs=1,
                                            patience=patience,
                                            balance_train_classes=balance_train_classes,
                                            test_weight_multiplier=test_weight_multiplier,
                                            show_progress_bar=show_progress_bar,
                                            verbose=verbose)

    return DetectronResult(calibration_record, test_record)


def detectron_dis_power(calibration_record: DetectronRecord,
                        test_record: DetectronRecord,
                        alpha=0.05,
                        max_ensemble_size=None):
    """
    Compute the discovery power of the detectron algorithm.
    :param calibration_record: (XGBDetectronRecord) the results of the calibration run
    :param test_record: (XGBDetectronRecord) the results of the test run
    :param alpha: (0.05) the significance level
    :param max_ensemble_size: (None) the maximum number of models in the ensemble to consider.
        If None, all models are considered.
    :return: the discovery power
    """
    cal_counts = calibration_record.counts(max_ensemble_size=max_ensemble_size)
    test_counts = test_record.counts(max_ensemble_size=max_ensemble_size)
    N = calibration_record.sample_size
    assert N == test_record.sample_size, 'The sample sizes of the calibration and test runs must be the same'

    fpr = (cal_counts <= np.arange(0, N + 2)[:, None]).mean(1)
    tpr = (test_counts <= np.arange(0, N + 2)[:, None]).mean(1)

    quantile = np.quantile(cal_counts, alpha)
    tpr_low = (test_counts < quantile).mean()
    tpr_high = (test_counts <= quantile).mean()

    fpr_low = (cal_counts < quantile).mean()
    fpr_high = (cal_counts <= quantile).mean()

    if fpr_high == fpr_low:
        tpr_at_alpha = tpr_high
    else:  # use linear interpolation if there is no threshold at alpha
        tpr_at_alpha = (tpr_high - tpr_low) / (fpr_high - fpr_low) * (alpha - fpr_low) + tpr_low

    return dict(power=tpr_at_alpha, auc=np.trapz(tpr, fpr), N=N)

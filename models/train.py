import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score

from data.gemini import GroupedDataConfig
from data.globals import LENGTH_OF_STAY, DISCHARGE_DISPOSITION

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'random_state': 0,
    'eval_metric': 'auc',
    'early_stopping_rounds': 10,
    'n_estimators': 100,
    'max_depth': 7,
}


class Tasks:
    @staticmethod
    def length_of_stay(days=3):
        def f(x):
            return x[LENGTH_OF_STAY] < days

        return f

    @staticmethod
    def mortality(days=7):
        def f(x):
            return x[LENGTH_OF_STAY] < days and x[DISCHARGE_DISPOSITION] == 1

        return f


class Trainer:
    def __init__(self,
                 data: dict[str, dict[str, tuple[np.array, np.array]]],
                 config: GroupedDataConfig,
                 params: dict = XGB_PARAMS,
                 verbose=True):
        self.data = data
        self.config = config
        self.hparams = params
        self.verbose = verbose
        self.models = {}
        self.groups = list(data.keys())
        self.results = pd.DataFrame(index=self.groups, columns=self.groups)

    def fit_on_group(self, group_name: str):
        X_train, y_train = self.data[group_name]['train']
        X_val, y_val = self.data[group_name]['val']
        model = self._fit_model(X_train, y_train, X_val, y_val)
        self.models[group_name] = model

    def get_model(self, group_name: str | int):
        assert group_name in self.models, f'Group {group_name} not found in trained models'
        return self.config.create_model_config(group_name=group_name, model=self.models[group_name],
                                               hparams=self.hparams)

    def _fit_model(self, X_train, y_train, X_val, y_val):
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
        clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **self.hparams)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=self.verbose)
        return clf

    @staticmethod
    def _evaluate_model(model, X_test, y_test, metric_func):
        y_pred = model.predict(X_test)
        return metric_func(y_test, y_pred)

    def fit(self):
        for group_name, group_data in self.data.items():
            X_train, y_train = group_data['train']
            X_val, y_val = group_data['val']
            model = self._fit_model(X_train, y_train, X_val, y_val)
            self.models[group_name] = model

    def evaluate(self, metric_func=roc_auc_score):
        for train_group, model in self.models.items():
            for test_group, group_data in self.data.items():
                X_test, y_test = group_data['test']
                metric = self._evaluate_model(model, X_test, y_test, metric_func)
                self.results.loc[train_group, test_group] = metric
        return self.results

    def evaluate_on_group(self, group_name: str, metric_func=roc_auc_score):
        self.results = pd.DataFrame(index=self.groups, columns=['data'])
        model = self.models[group_name]
        for test_group, group_data in self.data.items():
            X_test, y_test = group_data['test']
            metric = self._evaluate_model(model, X_test, y_test, metric_func)
            self.results.loc[test_group, 'data'] = metric
        return self.results

    def confusion_plot(self, grouper='hosp', task='mort-7'):
        ConfusionMatrixDisplay(self.results.to_numpy().astype(float), display_labels=self.results.index).plot()
        plt.xlabel('Test Group')
        plt.ylabel('Train Group')
        plt.savefig(f'results/{grouper}-{task}.png', dpi=300)
        plt.close()

    def ave_percent_drop(self, grouper='hosp', task='mort-7'):
        res = self.results.to_numpy().astype(float)
        res = (((np.diag(res)[:, None] - res.T) / np.diag(res)[:, None]) * 100).T
        ConfusionMatrixDisplay(res, display_labels=self.results.index).plot()
        plt.xlabel('Test Group')
        plt.ylabel('Train Group')
        plt.savefig(f'results/{grouper}-{task}-relative.png', dpi=300)
        plt.close()
        total = np.sum(res, axis=0)
        ave = total / (len(total) - 1)
        print(f'average drop per {grouper} for {task}: \n')
        print(ave)

    def time_dependent_auc(self, ax, separate_by, ref_result):
        ax.plot(self.results.index, self.results.to_numpy().astype(float), marker='o')
        # plot aggregate data line as reference
        # ax.plot(self.results.index, ref_result[:len(self.results.index)], linestyle='--', color='gray', marker='o')
        ax.set_xlabel('Time')
        ax.set_ylabel('AUC')
        ax.set_title(f'hospital {separate_by}')

        # Shading
        COVID = '2020H1'  # assuming 2020 starts the COVID era

        # Shading Pre-COVID
        ax.axvspan(self.results.index.min(), COVID, color='green', alpha=0.2, label='Pre-COVID')
        # Shading Post-COVID
        ax.axvspan(COVID, self.results.index.max(), color='red', alpha=0.2, label='COVID')

        ax.legend()

        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='x', labelsize=8)

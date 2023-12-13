from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons


class SyntheticData(pd.DataFrame):
    def __init__(self, n_samples=1000, alpha=10, beta=.5, random_state=0):
        np.random.seed(random_state)
        super().__init__(columns=['x1', 'x2', 'y', 'date'])

        self['date'] = pd.date_range(
            start=datetime(2019, 1, 1),
            end=datetime(2023, 12, 12),
            periods=n_samples
        )

        p_data = SyntheticData.p_data(n_samples=n_samples, random_state=random_state)
        q_data = SyntheticData.q_data(n_samples=n_samples)

        x = alpha * (np.linspace(0, 1, n_samples) - beta)
        dynamics = 1 - (np.tanh(x / 2) + 1) / 2
        mask = np.random.random(n_samples) < dynamics

        x1 = np.where(mask, p_data[0][:, 0], q_data[0][:, 0])
        x2 = np.where(mask, p_data[0][:, 1], q_data[0][:, 1])
        labels = np.where(mask, p_data[1], q_data[1])

        self['x1'] = x1
        self['x2'] = x2
        self['y'] = labels

    @staticmethod
    def labels(data):
        x1, x2 = data[:, 0], data[:, 1]
        return x2 > np.sin(x1)

    @staticmethod
    def p_data(n_samples=100, noise=0.15, random_state=0):
        data = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)[0]
        return data, SyntheticData.labels(data)

    @staticmethod
    def q_data(n_samples=100):
        q = np.random.randn(n_samples, 2) * np.array([[5, .25]])
        return q, SyntheticData.labels(q)

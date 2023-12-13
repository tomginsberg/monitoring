from abc import ABC, abstractmethod


class DatasetBase(ABC):

    @abstractmethod
    def train_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def val_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def test_dataset(self):
        raise NotImplementedError

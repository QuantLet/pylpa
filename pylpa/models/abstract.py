from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def my_abstract_method(self):

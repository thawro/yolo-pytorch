from abc import ABC, abstractmethod

from tqdm.auto import tqdm

from src.utils.utils import colorstr

from .results import BaseResult


class BaseValidator(ABC):
    prefix = colorstr("Validator: ")
    results: list[BaseResult]

    def __init__(self):
        self.results = []

    def update_results(self, results: list[BaseResult]):
        self.results.extend(results)

    def process_results(self):
        for result in tqdm(self.results, desc="Processing validation results"):
            result.set_preds()

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError()

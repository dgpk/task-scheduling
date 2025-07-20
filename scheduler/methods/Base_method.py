from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseMethod(ABC):
    """
    Abstrakcyjna klasa bazowa (interfejs) dla wszystkich metod harmonogramowania.
    """

    def __init__(self, machines_df: pd.DataFrame, tasks_df: pd.DataFrame, etc_matrix: np.ndarray):
        self.machines_df = machines_df
        self.tasks_df = tasks_df
        self.etc_matrix = etc_matrix
        self.kwargs = {}

    def set_params(self, **kwargs):
        """Ustawia parametry specyficzne dla danego uruchomienia, np. mutation_rate."""
        self.kwargs = kwargs

    @abstractmethod
    def initialize_population(self, pop_size: int):
        """Tworzy i zwraca populację początkową."""
        pass

    @abstractmethod
    def evaluate_population(self, population):
        """Ocenia całą populację."""
        pass

    @abstractmethod
    def crossover(self, population):
        """Wykonuje operację krzyżowania na populacji."""
        pass

    @abstractmethod
    def mutation(self, population):
        """Wykonuje operację mutacji na populacji."""
        pass

    @abstractmethod
    def get_tasks_on_machines_list(self, individual) -> list[list[int]]:
        """Konwertuje genotyp najlepszego osobnika na listę list zadań."""
        pass

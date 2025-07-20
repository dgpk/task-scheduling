from random import randint
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import scheduler.Common as Common

# POPRAWKA: Import z poprawnej nazwy pliku
from scheduler.methods.Base_method import BaseMethod


class MichiganMethod(BaseMethod):
    """Implementacja algorytmu Michigan jako konkretna strategia."""

    def __init__(self, machines_df: pd.DataFrame, tasks_df: pd.DataFrame, etc_matrix: np.ndarray):
        super().__init__(machines_df, tasks_df, etc_matrix)
        self.columns = ["task_" + str(i) for i in range(len(self.tasks_df))]

    # --- Implementacja metod abstrakcyjnych ---

    def initialize_population(self, pop_size: int):
        tasks_cpy = self.tasks_df.index.values.copy()
        np.random.shuffle(tasks_cpy)
        tasks_arr = np.full((len(self.machines_df), len(self.columns)), np.nan)
        j = -1
        for i, task_val in enumerate(tasks_cpy):
            machine_idx = i % len(self.machines_df)
            if machine_idx == 0:
                j += 1
            if j < len(self.columns):
                tasks_arr[machine_idx][j] = task_val
        return self._numpy_arr_to_dataframe(tasks_arr, self.machines_df.index.values)

    def evaluate_population(self, population):
        machine_exec_times = self._calculate_times(population)
        schedule_makespan = np.max(machine_exec_times[:, 1]) if machine_exec_times.size > 0 else 0
        total_schedule_energy = 0
        for row in machine_exec_times:
            machine_id, busy_time = int(row[0]), row[1]
            idle_time = max(0, schedule_makespan - busy_time)
            p_busy = self.machines_df.loc[machine_id]["P_busy"]
            p_idle = self.machines_df.loc[machine_id]["P_idle"]
            total_schedule_energy += (busy_time * p_busy) + (idle_time * p_idle)

        primary_score = (
            schedule_makespan
            if Common.scheduling_mode == Common.MAKESPAN_MODE
            else total_schedule_energy
        )
        secondary_score = (
            total_schedule_energy
            if Common.scheduling_mode == Common.MAKESPAN_MODE
            else schedule_makespan
        )
        return primary_score, population, secondary_score

    def crossover(self, population):
        top, bottom = self._split_population(population)
        num_pairs = min(len(top), len(bottom))
        for i in range(num_pairs):
            self._cross_pair(top.values[i], bottom.values[i])
        data = np.concatenate((top.values, bottom.values), axis=0)
        indexes = np.concatenate((top.index.values, bottom.index.values), axis=0)
        return self._numpy_arr_to_dataframe(data, indexes)

    def mutation(self, population):
        pm = self.kwargs.get("mutation_rate", 0.01)
        for machine_id in population.index:
            if np.random.uniform(0.0, 1.0) <= pm:
                self._shuffle_chromosome(population.loc[machine_id].values)
        return population

    def get_tasks_on_machines_list(self, individual):
        schedule_df = individual
        tasks_on_machines = []
        for machine_id in self.machines_df.index:
            tasks = [t for t in schedule_df.loc[machine_id].values if not np.isnan(t)]
            tasks_on_machines.append(tasks)
        return tasks_on_machines

    # --- Metody pomocnicze (dawne funkcje globalne) ---
    # ... (tutaj wklej wszystkie funkcje pomocnicze z Michigan.py jako metody prywatne, np. _get_last_task_index, _shuffle_chromosome, etc.)
    def _get_last_task_index(self, chromosome):
        for i, task_id in enumerate(chromosome):
            if np.isnan(task_id):
                return i - 1
        return len(chromosome) - 1

    def _shuffle_chromosome(self, chromosome):
        last_index = self._get_last_task_index(chromosome)
        if last_index >= 0:
            np.random.shuffle(chromosome[: last_index + 1])

    def _calculate_time_on_machine(self, tasks_ids, machine_id):
        time_sum = 0.0
        for task_id in tasks_ids:
            if np.isnan(task_id):
                break
            time_sum += self.etc_matrix[int(task_id)][int(machine_id)]
        return machine_id, time_sum

    def _calculate_times(self, population):
        tmp = np.zeros(shape=(len(population), 2), dtype=np.float64)
        for i, machine_id in enumerate(population.index.values):
            tmp[i] = self._calculate_time_on_machine(population.loc[machine_id].values, machine_id)
        return tmp

    def _numpy_arr_to_dataframe(self, data, ids):
        df = pd.DataFrame(data=data, index=ids, columns=self.columns)
        df.index.name = "machines"
        return df

    def _split_population(self, population):
        parts = np.array_split(population, 2)
        return shuffle(parts[0]), shuffle(parts[1])

    def _cross_pair(self, first, second):
        size_first = self._get_last_task_index(first) + 1
        size_second = self._get_last_task_index(second) + 1
        if size_first == 0 or size_second == 0:
            return
        cp1, cp2 = randint(0, size_first), randint(0, size_second)
        tmp1 = np.concatenate((first[:cp1], second[cp2:size_second]))
        tmp2 = np.concatenate((second[:cp2], first[cp1:size_first]))
        first.fill(np.nan)
        second.fill(np.nan)
        if len(tmp1) > 0:
            first[: len(tmp1)] = tmp1
        if len(tmp2) > 0:
            second[: len(tmp2)] = tmp2

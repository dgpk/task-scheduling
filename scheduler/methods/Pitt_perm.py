import numpy as np
from sklearn.utils import shuffle
import random
import math

import scheduler.Common as Common

# POPRAWKA: Import z pliku pisanego małymi literami
from scheduler.methods.Base_method import BaseMethod


class PittPermMethod(BaseMethod):
    """Implementacja algorytmu Pitt z reprezentacją permutacyjną."""

    def __init__(self, machines_df, tasks_df, etc_matrix):
        super().__init__(machines_df, tasks_df, etc_matrix)
        self.tasks_to_machines = self._map_tasks_to_possible_machines()

    # --- Implementacja metod abstrakcyjnych ---

    def initialize_population(self, pop_size: int):
        return [tuple(self._generate_individual()) for _ in range(pop_size)]

    def evaluate_population(self, population):
        best_score = float("inf")
        best_individual = None
        other_param_for_best = float("inf")

        for individual in population:
            score, other_param = self._get_metrics_for_individual(individual)
            if score < best_score:
                best_score = score
                best_individual = individual
                other_param_for_best = other_param

        return best_score, best_individual, other_param_for_best

    def crossover(self, population):
        shuffled_population = shuffle(population)
        new_population = []
        for i in range(0, len(shuffled_population) - 1, 2):
            parent1, parent2 = shuffled_population[i], shuffled_population[i + 1]
            child1_tasks, child2_tasks = self._ordered_crossover(parent1[0], parent2[0])
            new_population.append((child1_tasks, parent1[1]))
            new_population.append((child2_tasks, parent2[1]))
        if len(shuffled_population) % 2 == 1:
            new_population.append(shuffled_population[-1])
        return new_population

    def mutation(self, population):
        mutated_pop = []
        pm_s = self.kwargs.get("mutation_rate_s", 0.01)
        pm_t = self.kwargs.get("mutation_rate_t", 0.01)

        for ind in population:
            tasks_list, machines_list = list(ind[0]), list(ind[1])
            for i in range(len(tasks_list)):
                if random.random() < pm_s:
                    tasks_list, machines_list = self._swap_mutation((tasks_list, machines_list), i)
                if random.random() < pm_t:
                    tasks_list, machines_list = self._transposition_mutation(
                        (tasks_list, machines_list), i
                    )
            mutated_pop.append((tuple(tasks_list), tuple(machines_list)))
        return mutated_pop

    def get_tasks_on_machines_list(self, individual):
        tasks_chromosome, machines_chromosome = individual
        tasks_of_machines = []
        current_pos = 0
        for num_tasks in machines_chromosome:
            tasks_of_machines.append(list(tasks_chromosome[current_pos : current_pos + num_tasks]))
            current_pos += num_tasks
        return tasks_of_machines

    # --- Metody pomocnicze ---

    def _map_tasks_to_possible_machines(self):
        tasks_to_machines_map = {}
        for task_id in self.tasks_df.index:
            task = self.tasks_df.loc[task_id]
            possible_machines = [
                m_id
                for m_id in self.machines_df.index
                if Common.can_execute_task_on_machine(
                    self.machines_df.loc[m_id], task, Common.read_security_features()
                )
            ]
            tasks_to_machines_map[task_id] = possible_machines
        return tasks_to_machines_map

    def _generate_individual(self):
        num_tasks = len(self.tasks_df)
        num_machines = len(self.machines_df)
        base_tasks = math.floor(num_tasks / num_machines)
        machines_chromosome = [base_tasks] * num_machines
        for i in range(num_tasks % num_machines):
            machines_chromosome[i] += 1

        machines_to_tasks = {i: [] for i in self.machines_df.index}
        sorted_tasks = sorted(self.tasks_to_machines.items(), key=lambda item: len(item[1]))

        for task_id, possible_machines in sorted_tasks:
            random.shuffle(possible_machines)
            assigned = False
            for m_id in possible_machines:
                if len(machines_to_tasks[m_id]) < machines_chromosome[m_id]:
                    machines_to_tasks[m_id].append(task_id)
                    assigned = True
                    break
            if not assigned:  # Fallback if initial distribution is too strict
                m_id = random.choice(possible_machines)
                machines_to_tasks[m_id].append(task_id)

        tasks_chromosome = [
            task for tasks_list in machines_to_tasks.values() for task in tasks_list
        ]
        final_machines_chromosome = [len(tasks) for tasks in machines_to_tasks.values()]

        return tasks_chromosome, final_machines_chromosome

    def _get_machine_for_task_pos(self, individual, task_pos):
        counter = 0
        for machine_idx, num_tasks in enumerate(individual[1]):
            counter += num_tasks
            if task_pos < counter:
                return machine_idx
        return -1

    def _swap_mutation(self, individual, i):
        tasks, machines = individual
        num_tasks = len(tasks)
        j = random.randrange(num_tasks)

        machine_of_i = self._get_machine_for_task_pos(individual, i)
        machine_of_j = self._get_machine_for_task_pos(individual, j)

        if (
            machine_of_j in self.tasks_to_machines[tasks[i]]
            and machine_of_i in self.tasks_to_machines[tasks[j]]
        ):
            tasks[i], tasks[j] = tasks[j], tasks[i]
        return tasks, machines

    def _transposition_mutation(self, individual, i):
        tasks, machines = individual
        num_tasks = len(tasks)
        j = random.randrange(num_tasks)

        original_machine_idx = self._get_machine_for_task_pos(individual, i)
        target_machine_idx = self._get_machine_for_task_pos(individual, j)

        if target_machine_idx in self.tasks_to_machines[tasks[i]]:
            task_to_move = tasks.pop(i)
            tasks.insert(j, task_to_move)
            machines[original_machine_idx] -= 1
            machines[target_machine_idx] += 1
        return tasks, machines

    def _ordered_crossover(self, dad, mom):
        """
        POPRAWKA: Zastąpiono błędną implementację krzyżowania
        standardową i poprawną wersją Ordered Crossover (OX1).
        """
        size = len(dad)
        son, daughter = [-1] * size, [-1] * size

        start, end = sorted([random.randrange(size) for _ in range(2)])

        # --- Tworzenie potomka 1 (son) ---
        son[start : end + 1] = dad[start : end + 1]
        mom_genes_to_add = [gene for gene in mom if gene not in son]

        mom_genes_ptr = 0
        for i in range(size):
            if son[i] == -1:
                son[i] = mom_genes_to_add[mom_genes_ptr]
                mom_genes_ptr += 1

        # --- Tworzenie potomka 2 (daughter) ---
        daughter[start : end + 1] = mom[start : end + 1]
        dad_genes_to_add = [gene for gene in dad if gene not in daughter]

        dad_genes_ptr = 0
        for i in range(size):
            if daughter[i] == -1:
                daughter[i] = dad_genes_to_add[dad_genes_ptr]
                dad_genes_ptr += 1

        return tuple(son), tuple(daughter)

    def _get_metrics_for_individual(self, individual):
        tasks_chromosome, machines_chromosome = individual
        machine_times = [0.0] * len(self.machines_df)
        task_offset = 0
        for m_idx, num_tasks in enumerate(machines_chromosome):
            for i in range(num_tasks):
                task_id = tasks_chromosome[task_offset + i]
                machine_times[m_idx] += self.etc_matrix[task_id][m_idx]
            task_offset += num_tasks

        makespan = max(machine_times) if machine_times else 0
        total_energy = 0
        for m_idx, busy_time in enumerate(machine_times):
            idle_time = makespan - busy_time
            p_busy = self.machines_df.iloc[m_idx]["P_busy"]
            p_idle = self.machines_df.iloc[m_idx]["P_idle"]
            total_energy += (busy_time * p_busy) + (idle_time * p_idle)

        if Common.scheduling_mode == Common.ENERGY_MODE:
            return total_energy, makespan
        # POPRAWKA: Zawsze zwracaj krotkę (makespan, total_energy) w trybie makespan.
        return makespan, total_energy

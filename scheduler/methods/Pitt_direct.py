import random
import scheduler.Common as Common
# POPRAWKA: Import z poprawnej nazwy pliku
from scheduler.methods.Base_method import BaseMethod

class PittDirectMethod(BaseMethod):
    """Implementacja algorytmu Pitt z reprezentacją bezpośrednią."""

    def __init__(self, machines_df, tasks_df, etc_matrix):
        super().__init__(machines_df, tasks_df, etc_matrix)

    def initialize_population(self, pop_size: int):
        return [self._generate_one_individual() for _ in range(pop_size)]

    def evaluate_population(self, population):
        best_score = float('inf')
        best_individual = None
        other_param_for_best = float('inf')

        for individual in population:
            score, other_param = self._get_metrics_for_individual(individual)
            if score < best_score:
                best_score = score
                best_individual = individual
                other_param_for_best = other_param
        
        return best_score, best_individual, other_param_for_best

    def crossover(self, population):
        new_population = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = self._crossover_pair(parent1, parent2)
            new_population.extend([child1, child2])
        return new_population

    def mutation(self, population):
        pm = self.kwargs.get('mutation_rate', 0.01)
        return [self._mutate_individual(ind, pm) for ind in population]

    def get_tasks_on_machines_list(self, individual):
        num_machines = len(self.machines_df)
        tasks_on_machines = [[] for _ in range(num_machines)]
        for task_id, machine_id in enumerate(individual):
            tasks_on_machines[machine_id].append(task_id)
        return tasks_on_machines

    # --- Metody pomocnicze ---

    def _generate_one_individual(self):
        num_tasks = len(self.tasks_df)
        num_machines = len(self.machines_df)
        individual = [-1] * num_tasks
        
        # Ensure each machine gets at least one task if possible
        assigned_machines = list(range(num_machines))
        random.shuffle(assigned_machines)
        for i in range(min(num_tasks, num_machines)):
            individual[i] = assigned_machines[i]
        
        # Assign remaining tasks
        for i in range(num_machines, num_tasks):
            individual[i] = random.randrange(num_machines)
        
        random.shuffle(individual) # Shuffle to make it random
        
        # A simple validation/repair step could be added here if needed
        # to ensure all tasks can run on their assigned machines.
        # For now, we assume the random assignment is valid enough for a start.
        return individual

    def _crossover_pair(self, parent1, parent2):
        size = len(parent1)
        cp = random.randint(1, size - 1)
        child1 = parent1[:cp] + parent2[cp:]
        child2 = parent2[:cp] + parent1[cp:]
        return child1, child2

    def _mutate_individual(self, individual, pm):
        mutated_ind = list(individual)
        for i in range(len(mutated_ind)):
            if random.random() < pm:
                current_machine = mutated_ind[i]
                new_machine = random.randrange(len(self.machines_df))
                while new_machine == current_machine:
                    new_machine = random.randrange(len(self.machines_df))
                mutated_ind[i] = new_machine
        return mutated_ind

    def _get_metrics_for_individual(self, individual):
        machine_times = [0.0] * len(self.machines_df)
        for task_id, machine_id in enumerate(individual):
            machine_times[machine_id] += self.etc_matrix[task_id][machine_id]
        
        makespan = max(machine_times) if machine_times else 0
        total_energy = 0
        for m_idx, busy_time in enumerate(machine_times):
            idle_time = makespan - busy_time
            p_busy = self.machines_df.iloc[m_idx]['P_busy']
            p_idle = self.machines_df.iloc[m_idx]['P_idle']
            total_energy += (busy_time * p_busy) + (idle_time * p_idle)

        if Common.scheduling_mode == Common.ENERGY_MODE:
            return total_energy, makespan
        return makespan, total_energy
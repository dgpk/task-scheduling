import numpy as np
import pandas as pd
import csv
import scheduler.Common as Common
from scheduler.Output_formatter import ConsoleOutputFormatter, CsvOutputFormatter

# Importowanie wszystkich dostępnych klas metod
from scheduler.methods.Michigan import MichiganMethod
from scheduler.methods.Pitt_perm import PittPermMethod
from scheduler.methods.Pitt_direct import PittDirectMethod


class Scheduler:
    """
    Centralny menedżer harmonogramowania.
    Inicjalizuje i przechowuje wszystkie dostępne metody (strategie),
    a następnie uruchamia wybraną na żądanie.
    """

    def __init__(self):
        print("Initializing Scheduler and loading data...")
        # Wczytanie danych tylko raz
        self.features = Common.read_security_features()
        self.machines = Common.read_machines(self.features)
        self.tasks = Common.read_tasks(self.features)
        self.etc = Common.generate_etc_matrix(self.machines, self.tasks)

        # Inicjalizacja formaterów
        self.console_formatter = ConsoleOutputFormatter(Common)
        self.csv_formatter = CsvOutputFormatter(Common)

        # Utworzenie i przechowanie instancji wszystkich metod
        self.methods = {
            "michigan": MichiganMethod(self.machines, self.tasks, self.etc),
            "pitt_perm": PittPermMethod(self.machines, self.tasks, self.etc),
            "pitt_direct": PittDirectMethod(self.machines, self.tasks, self.etc),
        }
        print("Scheduler is ready.")

    def run_method(self, method_name: str, params: dict):
        """Uruchamia wybraną metodę z podanymi parametrami."""
        if method_name not in self.methods:
            raise ValueError(f"Method '{method_name}' is not recognized.")

        method = self.methods[method_name]

        # Ustawienie parametrów dla tego konkretnego uruchomienia
        kwargs = params.copy()
        pop_size = len(self.machines) if method_name == "michigan" else kwargs.pop("pop_size")
        epochs = kwargs.pop("epochs")
        method.set_params(**kwargs)

        print(f"\n--- Running {method.__class__.__name__} ---")

        # Główna pętla algorytmu genetycznego
        population = method.initialize_population(pop_size)
        best_score, best_individual, other_param = method.evaluate_population(population)

        mode_name = Common.scheduling_modes[Common.scheduling_mode]
        other_mode_name = Common.scheduling_modes[(Common.scheduling_mode + 1) % 2]
        print(f"Initial best {mode_name}: {best_score:.2f} | {other_mode_name}: {other_param:.2f}")

        for i in range(epochs):
            population = method.crossover(population)
            population = method.mutation(population)
            current_score, current_best_ind, current_other_param = method.evaluate_population(
                population
            )

            if current_score < best_score:
                best_score, best_individual, other_param = (
                    current_score,
                    current_best_ind,
                    current_other_param,
                )
                print(
                    f"[{i}] New best {mode_name}: {best_score:.2f} | {other_mode_name}: {other_param:.2f}"
                )

        print(
            f"\nOptimization finished. Final best {mode_name}: {best_score:.2f} | {other_mode_name}: {other_param:.2f}"
        )

        if best_individual is not None:
            self._report_results(method, best_individual, best_score, other_param)

    def _report_results(self, method, individual, score, other_param):
        final_makespan = score if Common.scheduling_mode == Common.MAKESPAN_MODE else other_param
        tasks_on_machines = method.get_tasks_on_machines_list(individual)
        max_tasks = max(len(tasks) for tasks in tasks_on_machines) if any(tasks_on_machines) else 0

        pseudo_df = self._create_pseudo_df(tasks_on_machines, max_tasks)

        print("\n" + self.console_formatter.format_header(pseudo_df, score, other_param))
        for i, machine_id in enumerate(self.machines.index):
            print(
                self.console_formatter.format_machine_row(
                    machine_id,
                    tasks_on_machines[i],
                    self.etc,
                    self.machines,
                    final_makespan,
                    max_tasks,
                )
            )
        print(self.console_formatter.format_footer())

        class_name = method.__class__.__name__.lower().replace("method", "")
        output_file = f"scheduler/results/output_{class_name}.csv"
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows(self.csv_formatter.format_header(pseudo_df, score, other_param))
            for i, machine_id in enumerate(self.machines.index):
                writer.writerow(
                    self.csv_formatter.format_machine_row(
                        machine_id,
                        tasks_on_machines[i],
                        self.etc,
                        self.machines,
                        final_makespan,
                        max_tasks,
                    )
                )

        summary_file = f"scheduler/results/result_{class_name}.csv"
        with open(summary_file, "a") as f:
            f.write(f"{score},{other_param}\n")

    def _create_pseudo_df(self, tasks_on_machines, max_tasks):
        data = {
            m_id: (tasks + [np.nan] * (max_tasks - len(tasks)))
            for m_id, tasks in zip(self.machines.index, tasks_on_machines)
        }
        return pd.DataFrame.from_dict(data, orient="index")

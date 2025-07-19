import numpy as np
import scheduler.Common as Common # Załóżmy, że Common jest dostępne

class OutputFormatter:
    def __init__(self, common_module):
        self.common = common_module

    def _get_task_details(self, task_id, machine_id, etc, machines_df):
        """Zwraca czas wykonania i energię dla zadania na maszynie."""
        exec_time = etc[int(task_id)][int(machine_id)]
        # machines_df.values[machine_id][2] to P_busy
        energy = exec_time * machines_df.values[machine_id][2]
        return exec_time, energy

    def _get_machine_summary_values(self, machine_id, tasks_on_machine_ids, etc, machines_df, max_overall_time):
        """Oblicza sumaryczny czas, energię (zajętą, bezczynną, całkowitą) dla maszyny."""
        time_sum = 0.0
        energy_sum_busy = 0.0
        for task_id in tasks_on_machine_ids:
            # pomijamy np.nan, które mogłyby tu trafić, jeśli lista nie jest przefiltrowana
            if np.isnan(task_id):
                continue
            exec_time, energy_busy_task = self._get_task_details(task_id, machine_id, etc, machines_df)
            time_sum += exec_time
            energy_sum_busy += energy_busy_task

        idle_time = max(0, max_overall_time - time_sum) # Czas bezczynności nie może być ujemny
        # machines_df.values[machine_id][3] to P_idle
        idle_energy = idle_time * machines_df.values[machine_id][3]
        total_energy = energy_sum_busy + idle_energy
        return time_sum, energy_sum_busy, idle_time, idle_energy, total_energy

    def format_header(self, population, best_score=None, other_param=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def format_machine_row(self, machine_id, tasks_on_machine_ids, etc, machines_df, max_overall_time, max_tasks_per_machine_in_pop):
        raise NotImplementedError("Subclasses should implement this method.")

    def format_footer(self, best_score=None, other_param=None):
        return None # Domyślnie brak stopki

    def get_max_tasks_for_population(self, population):
        """Pomocnicza funkcja do znalezienia maksymalnej liczby zadań na maszynę w populacji."""
        # Import get_last_task_index lokalnie, aby uniknąć problemów z zależnościami cyklicznymi
        # lub przenieś get_last_task_index do Common, jeśli jest używane globalnie.
        # Dla uproszczenia zakładamy, że jest dostępne w tym zakresie lub przekazane.
        # Z Michigan.py: from Michigan import get_last_task_index
        # Tutaj dla przykładu, jeśli nie można zaimportować:
        def get_last_task_index_local(chromosome):
            for i, task_id_val in enumerate(chromosome):
                if np.isnan(task_id_val):
                    return i - 1
            return len(chromosome) - 1

        max_tasks = 0
        if not population.empty:
            for i in range(len(population)):
                # Używamy +1, bo get_last_task_index zwraca indeks
                max_tasks = max(max_tasks, get_last_task_index_local(population.values[i]) + 1)
        return max_tasks

class ConsoleOutputFormatter(OutputFormatter):
    def format_header(self, population, best_score=None, other_param=None):
        header_lines = ["-----------------------------------------------------"]
        max_tasks_count = self.get_max_tasks_for_population(population)

        columns_str = format('Machine', '<8s') # Zmieniono na 'Machine' i wyrównanie
        for i in range(max_tasks_count):
            columns_str += format(f"Task {i}", '<18s') # Zwiększona szerokość dla czytelności

        summary_header = ""
        if self.common.output_mode == self.common.MAKESPAN_O_MODE:
            summary_header = format("Makespan", '>10s')
        elif self.common.output_mode == self.common.ENERGY_O_MODE:
            columns_str += format("Idle (E)", '<15s')
            summary_header = format("Total Energy", '>12s')
        elif self.common.output_mode == self.common.ALL_O_MODE:
            columns_str += format("Idle (T|E)", '<15s')
            summary_header = format("Makespan", '>10s') + format("Total Energy", '>12s')
        
        columns_str += " | " + summary_header
        header_lines.append(columns_str)
        header_lines.append("-" * len(columns_str)) # Linia dopasowana do długości nagłówka
        return "\n".join(header_lines)

    def format_machine_row(self, machine_id, tasks_on_machine_ids, etc, machines_df, max_overall_time, max_tasks_per_machine_in_pop):
        row_str = format(str(machine_id), '<8s') # Wyrównanie ID maszyny

        task_details_parts = []
        for i in range(max_tasks_per_machine_in_pop):
            if i < len(tasks_on_machine_ids):
                task_id = tasks_on_machine_ids[i]
                exec_time, energy = self._get_task_details(task_id, machine_id, etc, machines_df)
                
                display_val = ""
                if self.common.output_mode == self.common.MAKESPAN_O_MODE:
                    display_val = f"{int(task_id)} ({exec_time:.1f}t)"
                elif self.common.output_mode == self.common.ENERGY_O_MODE:
                    display_val = f"{int(task_id)} ({energy:.1f}e)"
                elif self.common.output_mode == self.common.ALL_O_MODE:
                    display_val = f"{int(task_id)} ({exec_time:.1f}t|{energy:.1f}e)"
                task_details_parts.append(format(display_val, '<18s'))
            else:
                task_details_parts.append(format('', '<18s')) # Puste miejsce dla zadań
        
        row_str += "".join(task_details_parts)

        time_sum, _, idle_time, idle_energy, total_energy = self._get_machine_summary_values(
            machine_id, tasks_on_machine_ids, etc, machines_df, max_overall_time
        )

        summary_str = " | "
        if self.common.output_mode == self.common.MAKESPAN_O_MODE:
            summary_str += format(f"{time_sum:.2f}", '>10s')
        elif self.common.output_mode == self.common.ENERGY_O_MODE:
            idle_display = f"{idle_time:.1f}t|{idle_energy:.1f}e" # Zmieniono dla jasności
            row_str += format(idle_display, '<15s')
            summary_str += format(f"{total_energy:.2f}", '>12s')
        elif self.common.output_mode == self.common.ALL_O_MODE:
            idle_display = f"{idle_time:.1f}t|{idle_energy:.1f}e"
            row_str += format(idle_display, '<15s')
            summary_str += format(f"{time_sum:.2f}", '>10s') + format(f"{total_energy:.2f}", '>12s')
            
        return row_str + summary_str

    def format_footer(self, best_score=None, other_param=None):
        return "-----------------------------------------------------"


class CsvOutputFormatter(OutputFormatter):
    def format_header(self, population, best_score=None, other_param=None):
        header_rows = []
        if best_score is not None:
            optimized_metric = self.common.scheduling_modes[self.common.scheduling_mode]
            header_rows.append(
                [f'{optimized_metric} optimized', f'{best_score:.2f}'.replace('.', ',')]
            )
            # Dodanie informacji o drugim parametrze, jeśli istnieje
            if other_param is not None:
                 other_metric = self.common.scheduling_modes[(self.common.scheduling_mode + 1) % 2]
                 header_rows.append([f'{other_metric}', f'{other_param:.2f}'.replace('.', ',')])


        max_tasks_count = self.get_max_tasks_for_population(population)
        task_header_row = ['MachineID']
        for i in range(max_tasks_count):
            task_header_row.append(f'Task_{i}_ID(Time|Energy)') # Bardziej opisowy nagłówek

        if self.common.output_mode == self.common.ENERGY_O_MODE or \
           self.common.output_mode == self.common.ALL_O_MODE:
            task_header_row.append('IdleTime')
            task_header_row.append('IdleEnergy')
        
        task_header_row.append('TotalMachineTime')
        task_header_row.append('TotalMachineEnergy')
        header_rows.append(task_header_row)
        return header_rows

    def format_machine_row(self, machine_id, tasks_on_machine_ids, etc, machines_df, max_overall_time, max_tasks_per_machine_in_pop):
        row_list = [str(machine_id)]
        
        for i in range(max_tasks_per_machine_in_pop):
            if i < len(tasks_on_machine_ids):
                task_id = tasks_on_machine_ids[i]
                exec_time, energy = self._get_task_details(task_id, machine_id, etc, machines_df)
                
                # Formatowanie zależy od output_mode, ale dla CSV warto dać obie wartości
                # lub dostosować do specyficznych potrzeb analizy CSV
                if self.common.output_mode == self.common.MAKESPAN_O_MODE:
                    row_list.append(f"{int(task_id)}({exec_time:.1f}t)".replace('.', ','))
                elif self.common.output_mode == self.common.ENERGY_O_MODE:
                    row_list.append(f"{int(task_id)}({energy:.1f}e)".replace('.', ','))
                elif self.common.output_mode == self.common.ALL_O_MODE:
                     row_list.append(f"{int(task_id)}({exec_time:.1f}t|{energy:.1f}e)".replace('.', ','))
            else:
                row_list.append('') # Puste dla brakujących zadań

        time_sum, _, idle_time, idle_energy, total_energy = self._get_machine_summary_values(
            machine_id, tasks_on_machine_ids, etc, machines_df, max_overall_time
        )

        if self.common.output_mode == self.common.ENERGY_O_MODE or \
           self.common.output_mode == self.common.ALL_O_MODE:
            row_list.append(f"{idle_time:.1f}".replace('.', ','))
            row_list.append(f"{idle_energy:.1f}".replace('.', ','))
        
        row_list.append(f"{time_sum:.2f}".replace('.', ','))
        row_list.append(f"{total_energy:.2f}".replace('.', ','))
            
        return row_list
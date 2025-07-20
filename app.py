import time
import scheduler.Common as Common
from scheduler.Scheduler import Scheduler


def main():
    # Utworzenie centralnego menedżera tylko raz
    main_scheduler = Scheduler()

    params = {
        "pop_size": 20,
        "epochs": 100,
        "mutation_rate": 0.02,
        "mutation_rate_s": 0.02,
        "mutation_rate_t": 0.02,
        "crossover_rate": 0.8,
    }

    method_map = {1: "pitt_perm", 2: "pitt_direct", 3: "michigan"}

    while True:
        try:
            menu_text = (
                "\nWhich algorithm would you like to use?\n"
                "1. Permutation-based Pitt algorithm\n"
                "2. Direct Pitt algorithm\n"
                "3. Michigan algorithm\n"
                "4. Switch scheduling mode (current: {})\n"
                "5. Switch output mode (current: {})\n"
                "6. Exit program\n"
            ).format(
                Common.scheduling_modes[Common.scheduling_mode],
                Common.output_modes[Common.output_mode],
            )
            userInput = int(input(menu_text))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if userInput in method_map:
            selected_method_name = method_map[userInput]
            # Wywołanie metody na istniejącej instancji schedulera
            main_scheduler.run_method(selected_method_name, params)
            print("--- Finished ---")
            time.sleep(2)
        elif userInput == 4:
            Common.scheduling_mode = (Common.scheduling_mode + 1) % 2
            print(f"Switched scheduling mode to: {Common.scheduling_modes[Common.scheduling_mode]}")
            time.sleep(1)
        elif userInput == 5:
            Common.output_mode = (Common.output_mode + 1) % 3
            print(f"Switched output mode to: {Common.output_modes[Common.output_mode]}")
            time.sleep(1)
        elif userInput == 6:
            break
        else:
            print("Wrong choice")
            time.sleep(1)


if __name__ == "__main__":
    main()

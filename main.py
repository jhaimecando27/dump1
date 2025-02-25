import sys
import time
import os

from statistics import mean, stdev

import config

from datetime import datetime
from algorithms.current import core
from algorithms.enhancements import obj1
from algorithms.enhancements import obj3

final_output = ""

current_timestamp = datetime.now().strftime("%m%d_%H%M%S")


def run_simulation():
    global final_output
    output = ""

    for n in range(len(config.pois)):
        for j in range(len(config.tenures)):

            list_soln_best: list[int] = []
            list_soln_best_tracker: list[list[int]] = []
            list_time: float = []

            for i in range(config.total_iter):

                s_time = time.perf_counter()
                soln_best, soln_best_tracker = core.tabu_search(
                    soln_init=config.soln_inits[config.pois[n]],
                    tabu_tenure=config.tenures[j],
                    iter_max=config.total_iter,
                )
                e_time = time.perf_counter()

                list_soln_best.append(soln_best)
                list_soln_best_tracker.append(soln_best_tracker)
                list_time.append(e_time - s_time)

                sys.stdout.write("\r\033[K")
                sys.stdout.write(
                    f"\r(STATUS) POI: {n + 1}/{len(config.pois)} | Tenure: {j+1}/{len(config.tenures)} |  run: {i+1}/{config.total_iter}"
                )
                sys.stdout.flush()

            soln_lst = []
            for run in list_soln_best_tracker:
                for soln in run:
                    soln_lst.append(soln)
            soln_lst.sort()

            list_soln_best.sort()

            # Calculate
            avg_soln = round(mean(soln_lst), 2)
            best_soln = round(min(soln_lst), 2)
            worst_soln = round(max(soln_lst), 2)

            # Relative differences and improvements
            dif_soln = round(((worst_soln - best_soln) / best_soln) * 100, 2)

            # Time statistics
            avg_time = round(mean(list_time), 2)

            # Additional statistics if needed
            std_dev = round(stdev(soln_lst), 2) if len(soln_lst) > 1 else 0

            # Store results
            output += f"POI: {config.pois[n]} | Tenure: {config.tenures[j]}\n"
            output += f"avg soln: {avg_soln}\n"
            output += f"dif: {dif_soln}\n"
            output += f"avg time: {avg_time}\n"
            output += f"std dev: {std_dev}\n"
            output += "================\n\n"

    print()
    print("\nFinished\n\n")
    print(output)

    final_output += "\n=====Result (Current)=====\n" + output


def run_simulation2():
    global final_output
    output = ""

    for n in range(len(config.pois)):
        for j in range(len(config.tenures)):

            list_soln_best: list[int] = []
            list_soln_best_tracker: list[list[int]] = []
            list_time: float = []

            for i in range(config.total_iter):

                s_time = time.perf_counter()
                soln_best, soln_best_tracker = obj1.tabu_search_with_perturbation(
                    soln_init=config.soln_inits[config.pois[n]],
                    tabu_tenure=config.tenures[j],
                    iter_max=config.total_iter,
                )
                e_time = time.perf_counter()

                list_soln_best.append(soln_best)
                list_soln_best_tracker.append(soln_best_tracker)
                list_time.append(e_time - s_time)

                sys.stdout.write("\r\033[K")
                sys.stdout.write(
                    f"\r(STATUS: Perturbation) POI: {n + 1}/{len(config.pois)} | Tenure: {j+1}/{len(config.tenures)} |  run: {i+1}/{config.total_iter}"
                )
                sys.stdout.flush()

            soln_lst = []
            for run in list_soln_best_tracker:
                for soln in run:
                    soln_lst.append(soln)
            soln_lst.sort()

            list_soln_best.sort()

            # Calculate
            avg_soln = round(mean(soln_lst), 2)
            best_soln = round(min(soln_lst), 2)
            worst_soln = round(max(soln_lst), 2)

            # Relative differences and improvements
            dif_soln = round(((worst_soln - best_soln) / best_soln) * 100, 2)

            # Time statistics
            avg_time = round(mean(list_time), 2)

            # Additional statistics if needed
            std_dev = round(stdev(soln_lst), 2) if len(soln_lst) > 1 else 0

            # Store results
            output += f"POI: {config.pois[n]} | Tenure: {config.tenures[j]}\n"
            output += f"avg soln: {avg_soln}\n"
            output += f"dif: {dif_soln}\n"
            output += f"avg time: {avg_time}\n"
            output += f"std dev: {std_dev}\n"
            output += "================\n\n"

    print()
    print("\nFinished")
    print(output)

    final_output += "\n=====Result (Perturbation)=====\n" + output


def run_simulation3():
    global final_output
    output = ""

    for n in range(len(config.pois)):

        list_soln_best: list[int] = []
        list_soln_best_tracker: list[list[int]] = []
        list_time: float = []

        for i in range(config.total_iter):

            s_time = time.perf_counter()
            soln_best, soln_best_tracker = obj3.ts_adaptive_tenure(
                soln_init=config.soln_inits[config.pois[n]],
                iter_max=config.total_iter,
            )
            e_time = time.perf_counter()

            list_soln_best.append(soln_best)
            list_soln_best_tracker.append(soln_best_tracker)
            list_time.append(e_time - s_time)

            sys.stdout.write("\r\033[K")
            sys.stdout.write(
                f"\r(STATUS: Adaptive Tenure) POI: {n + 1}/{len(config.pois)} | run: {i+1}/{config.total_iter}"
            )
            sys.stdout.flush()

        soln_lst = []
        for run in list_soln_best_tracker:
            for soln in run:
                soln_lst.append(soln)
        soln_lst.sort()

        list_soln_best.sort()

        # Calculate
        avg_soln = round(mean(soln_lst), 2)
        best_soln = round(min(soln_lst), 2)
        worst_soln = round(max(soln_lst), 2)

        # Relative differences and improvements
        dif_soln = round(((worst_soln - best_soln) / best_soln) * 100, 2)

        # Time statistics
        avg_time = round(mean(list_time), 2)

        # Additional statistics if needed
        std_dev = round(stdev(soln_lst), 2) if len(soln_lst) > 1 else 0

        # Store results
        output += f"POI: {config.pois[n]} \n"
        output += f"avg soln: {avg_soln}\n"
        output += f"dif: {dif_soln}\n"
        output += f"avg time: {avg_time}\n"
        output += f"std dev: {std_dev}\n"
        output += "================\n\n"

    print()
    print("\nFinished")
    print(output)

    final_output += "\n=====Result (Adaptive Tenure)=====\n" + output


if __name__ == "__main__":
    run_simulation3()
    run_simulation()
    run_simulation2()

    output_dir = os.path.join(os.path.dirname(__file__), "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    output_file_name = f"rf_{current_timestamp}.txt"
    output_file_path = os.path.join(output_dir, output_file_name)

    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(final_output)

import sys
import time
import os
from statistics import mean, stdev
import config
from datetime import datetime
from algorithms.current import core1, core2, core3
from algorithms.enhancements import hybrid1

final_output = ""
current_timestamp = datetime.now().strftime("%m%d_%H%M%S")


def run_core_simulation(core_module, core_name):
    global final_output
    output = ""

    for n in range(len(config.pois)):
        for j in range(len(config.tenures)):
            list_soln_best = []
            list_soln_best_tracker = []
            list_time = []

            for i in range(config.total_iter):
                sys.stdout.write("\r\033[K")
                sys.stdout.write(
                    f"\r(STATUS: {core_name}) POI: {n + 1}/{len(config.pois)} | Tenure: {j+1}/{len(config.tenures)} | run: {i+1}/{config.total_iter}"
                )
                sys.stdout.flush()

                s_time = time.perf_counter()

                if core_name == "core1":
                    soln_best, soln_best_tracker = core_module.tabu_search_genetic(
                        soln_init=config.soln_inits[config.pois[n]],
                        tabu_tenure=config.tenures[j],
                        iter_max=config.total_iter,
                    )
                else:
                    soln_best, soln_best_tracker = core_module.tabu_search(
                        soln_init=config.soln_inits[config.pois[n]],
                        tabu_tenure=config.tenures[j],
                        iter_max=config.total_iter,
                    )

                e_time = time.perf_counter()

                list_soln_best.append(soln_best)
                list_soln_best_tracker.append(soln_best_tracker)
                list_time.append(e_time - s_time)

            # Calculate statistics
            soln_lst = [soln for run in list_soln_best_tracker for soln in run]
            soln_lst.sort()

            avg_soln = round(mean(soln_lst), 2)
            best_soln = round(min(soln_lst), 2)
            worst_soln = round(max(soln_lst), 2)
            dif_soln = round(((worst_soln - best_soln) / best_soln) * 100, 2)
            avg_time = round(mean(list_time), 2)
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

    final_output += f"\n=====Result ({core_name})=====\n" + output


def run_hybrid_simulation():
    global final_output
    output = ""

    for n in range(len(config.pois)):
        list_soln_best = []
        list_soln_best_tracker = []
        list_time = []
        list_stagnant_best = []

        for i in range(config.total_iter):
            sys.stdout.write("\r\033[K")
            sys.stdout.write(
                f"\r(STATUS: Hybrid1) POI: {n + 1}/{len(config.pois)} | run: {i+1}/{config.total_iter}"
            )
            sys.stdout.flush()

            s_time = time.perf_counter()
            soln_best, soln_best_tracker, stagnant_best = hybrid1.search(
                soln_init=config.soln_inits[config.pois[n]],
                iter_max=config.total_iter,
            )
            e_time = time.perf_counter()

            list_soln_best.append(soln_best)
            list_soln_best_tracker.append(soln_best_tracker)
            list_time.append(e_time - s_time)
            list_stagnant_best.append(stagnant_best)

        # Calculate statistics
        soln_lst = [soln for run in list_soln_best_tracker for soln in run]
        soln_lst.sort()

        avg_soln = round(mean(soln_lst), 2)
        best_soln = round(min(soln_lst), 2)
        worst_soln = round(max(soln_lst), 2)
        stagnant_best = round(mean(list_stagnant_best), 2)
        dif_soln = round(((worst_soln - best_soln) / best_soln) * 100, 2)
        avg_time = round(mean(list_time), 2)
        std_dev = round(stdev(soln_lst), 2) if len(soln_lst) > 1 else 0

        # Store results
        output += f"POI: {config.pois[n]}\n"
        output += f"avg soln: {avg_soln}\n"
        output += f"dif: {dif_soln}\n"
        output += f"avg time: {avg_time}\n"
        output += f"std dev: {std_dev}\n"
        output += f"stagnant best: {stagnant_best}\n"
        output += "================\n\n"

    print()
    print("\nFinished\n\n")
    print(output)

    final_output += "\n=====Result (Enhanced)=====\n" + output


if __name__ == "__main__":
    run_core_simulation(core1, "core1")
    run_core_simulation(core2, "core2")
    run_core_simulation(core3, "core3")
    run_hybrid_simulation()

    # Save results to file
    output_dir = os.path.join(os.path.dirname(__file__), "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    output_file_name = f"rff_{current_timestamp}.txt"
    output_file_path = os.path.join(output_dir, output_file_name)

    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(final_output)

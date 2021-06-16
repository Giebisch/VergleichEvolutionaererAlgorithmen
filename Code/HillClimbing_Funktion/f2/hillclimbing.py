import argparse
import random
import os
from datetime import datetime
import numpy

args = None
EVALUATIONS = 0

### Hill climbing

def get_fitness(genes):
    """Get fitness for given genes

    Args:
        genes

    Returns:
        value [float]: fitness
    """
    value = get_function_value(genes)
    # make sure, that fitness = 0 doesnt lead to division by zero
    return 1/(1 + abs(value))

def get_function_value(genes):
    """Returns the value of the optimization problem

    Args:
        genes

    Returns:
        value [float]
    """
    global EVALUATIONS
    EVALUATIONS += 1
    value = 100 * (genes[1] - genes[0] ** 2)**2 + (1 - genes[0])**2
    return value

def gaussian_mutation(chromosome):
    """Uses gaussian mutation to create

    Args:
        population ([chromosomes])

    Returns:
        population: mutated Population
    """
    new_population = []
    for _ in range(57):
        new_solution = []
        for gene in chromosome:
            gene = numpy.random.normal(gene, 0.3)
            new_solution.append(gene)
        new_population.append(new_solution)
    return new_population

def get_best_solution(population):
    return sorted(population, key= lambda x: x.fitness, reverse=True)[0]

def main():
    """Creates initial population and then starts the genetic algorithm.
    Solutions will be saved in a file.
    """
    fitness_progress = []
    start_time = datetime.now()

    # create random first solution
    best_solution = [random.uniform(args.min, args.max) for _ in range(2)]

    # save fitness of initial population
    fitness_progress.append(get_function_value(best_solution))

    # generations
    for _ in range(0, 500):
        # create new solutions and pick the best one
        solution = sorted(gaussian_mutation(best_solution), key= lambda x: get_fitness(x), reverse=True)[0]

        if get_function_value(solution) < get_function_value(best_solution):
            best_solution = solution

        # track fitness progress
        fitness_progress.append(get_function_value(best_solution))

        # end if runtime > 20 seconds
        if (datetime.now() - start_time).total_seconds() > 20:
            break

    print(best_solution)
    print(f"Anzahl Evaluationen: {str(EVALUATIONS)}")

    timestamp = datetime.now().strftime("%H_%M_%S")

    # export csv file with fitness progress
    with open(os.path.join(args.folder, f"data_{timestamp}.csv"), "w") as file_out:
        file_out.write("Generations\tFunctionValue\n")
        for i, fitness in enumerate(fitness_progress):
            file_out.write(f"{str(i)}\t{fitness:.20f}\n")

    print(f"min = {str(get_function_value(best_solution))}")

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description='Hill climbing in Python \
        für das Optimieren einer Funktion')

    # upper and lower bound
    parser.add_argument('min', type=float, help="Set lower bound for search space")
    parser.add_argument('max', type=float, help="Set upper bound for search space")

    parser.add_argument('--folder', type=str, help='Specify output folder for the csvs',\
        default="")

    args = parser.parse_args()

    main()

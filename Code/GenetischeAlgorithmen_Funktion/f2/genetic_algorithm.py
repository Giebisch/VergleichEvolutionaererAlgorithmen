import argparse
import random
import os
from datetime import datetime
import numpy

args = None
EVALUATIONS = 0

### Genetic Algorithm

class Chromosome:
    """Class to save the fitness of the chromosomes
    """
    def __init__(self, genes):
        self.genes = genes
        self.fitness = get_fitness(self.genes)

def create_population(popsize, length):
    """Creates a new population with the size popsize and
    length-many genes

    Args:
        popsize: size of population
        length: number of genes per chromosome

    Returns:
        population
    """
    population = []
    for _ in range(0, popsize):
        new_chromosome = [random.uniform(args.min, args.max) for _ in range(length)]
        population.append(Chromosome(new_chromosome))
    return population

def tournament_select_chromosomes(population):
    """Uses tournament selection to select chromosomes out of
    the population

    Args:
        population

    Returns:
        [chromosomes]: selected chromosomes
    """
    new_population = []
    for _ in population:
        random_choice = random.choices(population, k=2)
        best_chromosome = sorted(random_choice, key= lambda x: x.fitness, reverse=True)[0]
        new_population.append(best_chromosome)

    return new_population

def rank_based_select_chromosomes(population):
    """Uses rank based selection to select chromosomes out of
    the population

    Args:
        population

    Returns:
        [chromosomes]: selected chromosomes
    """
    sorted_chromosomes = sorted(population, key= lambda x: x.fitness, reverse=True)

    weight = [i**2 for i in range(len(population), 0, -1)]

    # selects amount-many chromosomes based on their weight
    # the higher the weight the better the chances to be selected
    # It's possible to get duplicates
    chromosomes = random.choices(sorted_chromosomes, weights=weight, k=len(population))

    return chromosomes

def select_chromosomes(population):
    """Decides, which selection method will be used

    Args:
        population ([chromosomes])
        amount [int]: how many chromosomes will be selected

    Raises:
        Exception: Invalid selection selection

    Returns:
        [chromosomes]: selected chromosomes
    """
    # tournament selection
    if args.selection == 1:
        return tournament_select_chromosomes(population)
    # rank based selection
    elif args.selection == 2:
        return rank_based_select_chromosomes(population)
    else:
        raise Exception("selection argument has to be either 1 or 2")

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
    # value = 100 * (genes[0] ** 2 - genes[1])**2 + (1 - genes[0])**2
    value = 100 * (genes[1] - genes[0] ** 2)**2 + (1 - genes[0])**2
    return value

def two_point_crossover(population):
    """Uses two point crossover to create a new population, based
    on given probability args.crossover_p

    Args:
        population ([chromosomes])

    Returns:
        new population
    """
    new_population = []
    while len(population) > 1:
        parents = random.sample(population, k=2)
        if random.random() < args.crossover_p:
            parent1, parent2 = parents[0].genes, parents[1].genes

            # get the two crossover points
            index1 = random.randint(0, len(parent1) - 1)
            index2 = random.randint(0, len(parent2) - 1)

            if index1 > index2:
                index1, index2 = index2, index1

            child = parent1[0:index1] + parent2[index1:index2] + parent1[index2:]
            child2 = parent2[0:index1] + parent1[index1:index2] + parent2[index2:]

            new_population.append(Chromosome(child))
            new_population.append(Chromosome(child2))
        else:
        # add parents without crossover
            new_population.extend(parents)
        population.remove(parents[0])
        population.remove(parents[1])

    # if len(population) was odd, one last chromosome will be added without crossover
    if population:
        new_population.append(population[0])

    return new_population

def uniform_crossover(population):
    """Uses uniform crossover to create a new population, based
    on given probability args.crossover_p

    Args:
        population ([chromosomes])

    Returns:
        new population
    """
    new_population = []
    while len(population) > 1:
        parents = random.sample(population, k=2)
        if random.random() < args.crossover_p:
            parent1, parent2 = parents[0].genes, parents[1].genes

            child1 = []
            child2 = []

            for child in [child1, child2]:
                for i,_ in enumerate(parent1):
                    if random.randint(0, 1) == 0:
                        child.append(parent1[i])
                    else:
                        child.append(parent2[i])

            new_population.append(Chromosome(child1))
            new_population.append(Chromosome(child2))
        else:
        # add parents without crossover
            new_population.extend(parents)
        population.remove(parents[0])
        population.remove(parents[1])

    # if len(population) was odd, one last chromosome will be added without crossover
    if population:
        new_population.append(population[0])

    return new_population

def crossover(population):
    """Decides, which crossover method will be used

    Args:
        population ([chromosomes])

    Raises:
        Exception: Invalid crossover selection

    Returns:
        population: population
    """
    # two point crossover
    if args.crossover == 1:
        return two_point_crossover(population)
    # uniform crossover
    elif args.crossover == 2:
        return uniform_crossover(population)
    else:
        raise Exception("selection argument has to be either 1 or 2")

def uniform_mutation(population):
    """Uses uniform mutation to mutate a whole population, based
    on given probability args.mutation_p

    Args:
        population ([chromosomes])

    Returns:
        population: mutated Population
    """
    new_population = []
    for chromosome in population:
        new_chromosome = []
        for gene in chromosome.genes:
            if random.random() < args.mutation_p:
                gene = random.uniform(args.min, args.max)
            new_chromosome.append(gene)
        new_population.append(Chromosome(new_chromosome))
    return new_population

def gaussian_mutation(population):
    """Uses gaussian mutation to mutate a whole population, based
    on given probability args.mutation_p

    Args:
        population ([chromosomes])

    Returns:
        population: mutated Population
    """
    new_population = []
    for chromosome in population:
        new_chromosome = []
        for gene in chromosome.genes:
            if random.random() < args.mutation_p:
                gene = numpy.random.normal(gene, 0.3)
            new_chromosome.append(gene)
        new_population.append(Chromosome(new_chromosome))
    return new_population

def mutate(population):
    """Decides, which mutation method will be used

    Args:
        population ([chromosomes])

    Raises:
        Exception: Invalid mutation selection

    Returns:
        population: mutated population
    """
    # swap mutation
    if args.mutation == 1:
        return uniform_mutation(population)
    # gaussian mutation
    elif args.mutation == 2:
        return gaussian_mutation(population)
    else:
        raise Exception("mutation argument has to be either 1 or 2")

def get_best_solution(population):
    return sorted(population, key= lambda x: x.fitness, reverse=True)[0]

def main():
    """Creates initial population and then starts the genetic algorithm.
    Solutions will be saved in a file.
    """
    fitness_progress = []
    start_time = datetime.now()

    # create a random population
    population = create_population(30, 2)
    # save fitness of initial population
    fitness_progress.append(get_function_value(get_best_solution(population).genes))

    # generations
    for _ in range(0, 500):
        # randomly select new population influenced on their fitness

        selection = select_chromosomes(population)

        # breed random chromosomes
        breed = crossover(selection)

        # mutate some of the chromosomes
        population = mutate(breed)

        # track fitness progress
        fitness_progress.append(get_function_value(get_best_solution(population).genes))

        # end if runtime > 20 seconds
        if (datetime.now() - start_time).total_seconds() > 20:
            break

    best_solution = get_best_solution(population)
    print(best_solution.genes)
    print(f"Anzahl Evaluationen: {str(EVALUATIONS)}")

    timestamp = datetime.now().strftime("%H_%M_%S")

    # export csv file with fitness progress
    with open(os.path.join(args.folder, f"data_genetic_algorithm_{timestamp}.csv"), "w") as file_out:
        file_out.write("Generations\tFunctionValue\n")
        for i, fitness in enumerate(fitness_progress):
            file_out.write(f"{str(i)}\t{fitness:.20f}\n")

    print(f"Fitness = {str(best_solution.fitness)} | crossover_p = {str(args.crossover_p)} \
        | mutation_p = {str(args.mutation_p)}")

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description='Genetischer Algorithmus in Python \
        für das Optimieren einer Funktion')

    # change probabilities
    parser.add_argument('crossover_p', type=float, help='Probability of crossover')
    parser.add_argument('mutation_p', type=float, help='Probability of mutation')

    # change selection, crossover, mutation method
    parser.add_argument('selection', type=int, help="1 for tournament selection, \
        2 for rank based selection")
    parser.add_argument('crossover', type=int, help="1 for two point crossover, 2 for \
        uniform crossover")
    parser.add_argument('mutation', type=int, help="1 for uniform mutation, 2 for gaussian\
        mutation")

    # upper and lower bound
    parser.add_argument('min', type=float, help="Set lower bound for search space")
    parser.add_argument('max', type=float, help="Set upper bound for search space")

    parser.add_argument('--folder', type=str, help='Specify output folder for the csvs',\
        default="")

    args = parser.parse_args()

    main()

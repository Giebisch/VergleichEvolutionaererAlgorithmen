import argparse
import random
import os
from datetime import datetime
import numpy
import math

CITIES = []
args = None
EVALUATIONS = 0

# generates random number for the mutation of the strategy parameters
N_0_1 = numpy.random.normal(0, 1)

### General

def get_cities(input_file):
    global CITIES
    with open(input_file) as file:
        # skip first 6 lines / header
        lines = file.readlines()[6:-1]
        for _, line in enumerate(lines):
            tline = list(map(int, line.split(" ")))
            CITIES.append((tline[1], tline[2]))
    return CITIES

### Evolution strategy

class Chromosome:
    """Class to save the fitness and strategy parameters of the chromosomes
    """
    def __init__(self, route, sparams):
        self.route = route
        # in case of ES for TSP one strategy parameter per chromosome is sufficient
        self.sparams = sparams
        self.fitness = get_fitness(self.route)

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
        # create list of integers representing the cities
        genes = list(range(1, length + 1))
        # randomize order
        random.shuffle(genes)
        # set starting mutation step size
        sparams = 2
        population.append(Chromosome(genes, sparams))
    return population

def select(mutated_breed, population):
    """Decides, if either (mu,lambda)-ES or (mu+lambda)-ES will be used

    Args:
        mutated_breed: lambda-many mutants
        population: old population, used for (mu+lambda)-ES

    Raises:
        Exception: Invalid selection

    Returns:
        [chromosomes]: selected chromosomes
    """
    # mu, lambda
    if args.selection == 1:
        return sorted(mutated_breed, key= lambda x: x.fitness, reverse=True)[:args.mu]
    # mu + lambda
    elif args.selection == 2:
        union = mutated_breed + population
        return sorted(union, key= lambda x: x.fitness, reverse=True)[:args.mu]
    else:
        raise Exception("selection argument has to be either 1 or 2")

def get_fitness(route):
    """Get fitness (inverse of distance) for given chromosome

    Args:
        route

    Returns:
        value [float]: fitness
    """
    global EVALUATIONS
    EVALUATIONS += 1

    def get_distance(__city1, __city2):
        # returns euclidean distance
        return math.sqrt( (__city2[0] - __city1[0])**2 + (__city2[1] - __city1[1])**2 )

    distance = 0
    for i in range(0, len(route) - 1):
        city1 = CITIES[route[i] - 1]
        city2 = CITIES[route[i+1] - 1]
        distance += get_distance(city1, city2)
    # add distance from last to first city (round trip)
    distance += get_distance(CITIES[route[-1] - 1], CITIES[route[0] - 1])

    return 1/distance

def order_crossover(population):
    """Uses order crossover to create a new population

    Args:
        population ([chromosomes])

    Returns:
        new population
    """
    new_population = []
    while len(new_population) < args.lam:
        # select two parents, can't be the same
        parent1, parent2 = random.sample(population, k=2)

        # get the two crossover points
        index1 = random.randint(0, len(parent1.route) - 1)
        index2 = random.randint(0, len(parent2.route) - 1)

        if index1 > index2:
            index1, index2 = index2, index1

        ## child 1
        saved_part = parent1.route[index1:index2 + 1]
        # fill with the cities that are left (ordered)
        child_route = [gene for gene in parent2.route if gene not in saved_part]
        # insert saved_part at index1
        child_route[index1:index1] = saved_part

        # create child's strategy parameter
        child_sparams = numpy.mean([parent1.sparams, parent2.sparams])

        new_population.append(Chromosome(child_route, child_sparams))

    return new_population

def partially_mapped_crossover(population):
    """Uses partially mapped crossover to create a new population

    Args:
        population ([chromosomes])

    Returns:
        new population
    """
    def get_mapping(gene):
        """Changes genes according to the dictionary, loops until
        nothing changes anymore
        """
        t = mapping[gene]
        while (x := mapping[t]) != t:
            t = x
        return t

    new_population = []
    while len(new_population) < args.lam:
        parent1, parent2 = random.sample(population, k=2)

        # get the two crossover points
        index1 = random.randint(0, len(parent1.route) - 1)
        index2 = random.randint(0, len(parent2.route) - 1)

        if index1 > index2:
            index1, index2 = index2, index1

        mapping_section1 = parent1.route[index1:index2 + 1]
        mapping_section2 = parent2.route[index1:index2 + 1]

        mapping = dict()
        # standard mapping
        for i in range(1, len(parent1.route) + 1):
            mapping[i] = i
        # changing mapping from mapping_section1 to mapping_section2
        for i, gene in enumerate(mapping_section1):
            mapping[gene] = mapping_section2[i]

        child_route = [get_mapping(gene) for gene in parent2.route if gene not in mapping_section2]
        # insert saved_part at index1
        child_route[index1:index1] = mapping_section1

        # create child's strategy parameter
        child_sparams = numpy.mean([parent1.sparams, parent2.sparams])
        new_population.append(Chromosome(child_route, child_sparams))

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
    # order crossover
    if args.crossover == 1:
        return order_crossover(population)
    # partially mapped crossover
    elif args.crossover == 2:
        return partially_mapped_crossover(population)
    else:
        raise Exception("selection argument has to be either 1 or 2")

def mutate_sparam(sparam):
    """Calculates new strategy paramter

    Args:
        sparam: old strategy parameter

    Returns:
        new_sparam: new strategy parameter
    """
    tau_1 = 0.1
    tau_2 = 0.2
    min_svalue = 0.5
    # maximum step size: < 5% of # cities
    max_svalue = 5

    new_sparam = sparam * math.exp(tau_1 * N_0_1 + tau_2 * numpy.random.normal(0, 1))
    if new_sparam < min_svalue: new_sparam = min_svalue
    if new_sparam > max_svalue: new_sparam = max_svalue

    return new_sparam

def swap_mutate(population):
    """Uses swap mutation to mutate a whole population

    Args:
        population ([chromosomes])

    Returns:
        population: mutated Population
    """
    mutated_population = []
    for chromosome in population:
        route = chromosome.route
        # do mutation <strategy-paramter>-times
        for _ in range(round(chromosome.sparams)):
            # randomly choose 2 genes
            gene_1 = random.randint(0, len(route) - 1)
            gene_2 = random.randint(0, len(route) - 1)
            # swap the 2 genes
            route[gene_1], route[gene_2] = route[gene_2], route[gene_1]
        new_sparams = mutate_sparam(chromosome.sparams)    
        mutated_population.append(Chromosome(route, new_sparams))
    return mutated_population

def insertion_mutate(population):
    """Uses insertion mutation to mutate a whole population

    Args:
        population ([chromosomes])

    Returns:
        population: mutated Population
    """
    mutated_population = []
    for chromosome in population:
        route = chromosome.route
        # do mutation <strategy-paramter>-times
        for _ in range(round(chromosome.sparams)):
            # randomly choose index for gene
            index_pick = random.randint(0, len(route) - 1)
            index_new = random.randint(0, len(route) - 2)
            # extract gene
            chosen_gene = route.pop(index_pick)
            # insert in new position
            route.insert(index_new, chosen_gene)
        new_sparam = mutate_sparam(chromosome.sparams)
        mutated_population.append(Chromosome(route, new_sparam))
    return mutated_population

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
        return swap_mutate(population)
    # insertion mutation
    elif args.mutation == 2:
        return insertion_mutate(population)
    else:
        raise Exception("mutation argument has to be either 1 or 2")

def get_best_solution(population):
    return sorted(population, key= lambda x: x.fitness, reverse=True)[0]

def main():
    """Creates initial population and then starts the evolutionary strategy
    Solutions will be saved in a file.
    """
    fitness_progress = []
    start_time = datetime.now()

    # get city coordinates from .tsp file, specified as a CLI argument
    get_cities(args.input_file)

    # create a random population
    population = create_population(args.mu, len(CITIES))

    # save fitness of initial population
    fitness_progress.append(1/(get_best_solution(population).fitness))

    # generations
    for _ in range(0, 2000):

        # breed chromosomes
        breed = crossover(population)

        # mutate the chromosomes
        mutated_breed = mutate(breed)

        # select new population based on their fitness
        population = select(mutated_breed, population)
        
        # track fitness progress
        fitness_progress.append(1/(get_best_solution(population).fitness))

        # end if runtime > 200 seconds
        if (datetime.now() - start_time).total_seconds() > 200:
            break

    best_solution = get_best_solution(population)
    print(best_solution.route)
    print(best_solution.sparams)
    print(f"Anzahl Evaluationen: {str(EVALUATIONS)}")

    timestamp = datetime.now().strftime("%H_%M_%S")

    # export csv file with fitness progress
    with open(os.path.join(args.folder, f"data_genetic_algorithm_{timestamp}.csv"), "w") as file_out:
        file_out.write("Generations\tFunctionValue\n")
        for i, fitness in enumerate(fitness_progress):
            file_out.write(f"{str(i)}\t{fitness:.20f}\n")

    print(f"min = {1/best_solution.fitness:.20f} | mu = {str(args.mu)} \
        | lambda = {str(args.lam)}")

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description='Evolutionsstrategien in Python \
        für das TSP')

    parser.add_argument('input_file', type=str,
                        help='Specify input file (in .tsp format)')

    # change probabilities
    parser.add_argument('mu', type=int, help='Set µ (mu)')
    parser.add_argument('lam', type=int, help='Set λ (lambda)')

    # change selection, crossover, mutation method
    parser.add_argument('selection', type=int, help="1 for (mu,lambda), \
        2 for (mu+lambda)")
    parser.add_argument('crossover', type=int, help="1 for order crossover, 2 for \
        partially mapped crossover")
    parser.add_argument('mutation', type=int, help="1 for swap mutate, 2 for\
        insertion_mutate")

    parser.add_argument('--folder', type=str, help='Specify output folder for the csvs',\
        default="")

    args = parser.parse_args()

    main()

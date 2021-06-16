import argparse
import math
import random
import os
from datetime import datetime


CITIES = []
EVALUATIONS = 0
args = None

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

### Genetic Algorithm

class Chromosome:
    """Class to save the route and fitness of the chromosomes
    """
    # this class is used to save the chromosomes fitness
    def __init__(self, route):
        self.route = route
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
        chromosome = list(range(1, length + 1))
        # randomize order
        random.shuffle(chromosome)
        population.append(Chromosome(chromosome))
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
        random_choice = random.sample(population, k=5)
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

def get_fitness(chromosome):
    """Get fitness (inverse of distance) for given chromosome

    Args:
        chromosome

    Returns:
        value [float]: fitness
    """
    global EVALUATIONS
    EVALUATIONS += 1

    def get_distance(__city1, __city2):
        # returns euclidean distance
        return math.sqrt( (__city2[0] - __city1[0])**2 + (__city2[1] - __city1[1])**2 )

    distance = 0
    for i in range(0, len(chromosome) - 1):
        city1 = CITIES[chromosome[i] - 1]
        city2 = CITIES[chromosome[i+1] - 1]
        distance += get_distance(city1, city2)
    # add distance from last to first city (round trip)
    distance += get_distance(CITIES[chromosome[-1] - 1], CITIES[chromosome[0] - 1])

    return 1/distance

def order_crossover(population):
    """Uses order crossover to create a new population, based
    on given probability args.crossover_p

    Args:
        population ([chromosomes])

    Returns:
        new population
    """
    new_population = []
    while len(population) > 1:
        # select two parents, can't be the same
        parents = random.sample(population, k=2)
        if random.random() < args.crossover_p:
            parent1, parent2 = parents[0].route, parents[1].route

            # get the two crossover points
            index1 = random.randint(0, len(parent1) - 1)
            index2 = random.randint(0, len(parent1) - 1)

            if index1 > index2:
                index1, index2 = index2, index1

            ## child 1
            saved_part = parent1[index1:index2 + 1]
            # fill with the cities that are left (ordered)
            child = [gene for gene in parent2 if gene not in saved_part]
            # insert saved_part at index1
            child[index1:index1] = saved_part

            ## child 2
            saved_part2 = parent2[index1:index2 + 1]
            # fill with the cities that are left (ordered)
            child2 = [gene for gene in parent1 if gene not in saved_part2]
            # insert saved_part2 at index1
            child2[index1:index1] = saved_part2

            new_population.append(Chromosome(child))
            new_population.append(Chromosome(child2))
        else:
            # add parents without crossover
            new_population.extend(parents)
        population.remove(parents[0])
        population.remove(parents[1])

    # if len(population) was odd, last chromosome will be added without crossover
    if population:
        new_population.append(population[0])

    return new_population

def partially_mapped_crossover(population):
    """Uses partially mapped crossover to create a new population, based
    on given probability args.crossover_p

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
    while len(population) > 1:
        parents = random.sample(population, k=2)
        if random.random() < args.crossover_p:
            parent1, parent2 = parents[0].route, parents[1].route

            # get the two crossover points
            index1 = random.randint(0, len(parent1) - 1)
            index2 = random.randint(0, len(parent2) - 1)

            if index1 > index2:
                index1, index2 = index2, index1

            mapping_section1 = parent1[index1:index2 + 1]
            mapping_section2 = parent2[index1:index2 + 1]

            ## child 1
            mapping = dict()
            # standard mapping
            for i in range(1, len(parent1) + 1):
                mapping[i] = i
            # changing mapping from mapping_section1 to mapping_section2
            for i, gene in enumerate(mapping_section1):
                mapping[gene] = mapping_section2[i]

            child = [get_mapping(gene) for gene in parent2 if gene not in mapping_section2]
            # insert saved_part at index1
            child[index1:index1] = mapping_section1

            ## child 2
            # standard mapping
            for i in range(1, len(parent1) + 1):
                mapping[i] = i
            # changing mapping from mapping_section2 to mapping_section1
            for i, gene in enumerate(mapping_section2):
                mapping[gene] = mapping_section1[i]

            child2 = [get_mapping(gene) for gene in parent1 if gene not in mapping_section1]
            # insert saved_part at index1
            child2[index1:index1] = mapping_section2

            new_population.append(Chromosome(child))
            new_population.append(Chromosome(child2))
        else:
            # add parents without crossover
            new_population.extend(parents)
        population.remove(parents[0])
        population.remove(parents[1])

    # if len(population) was odd, last chromosome will be added without crossover
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
    # order crossover (OX1)
    if args.crossover == 1:
        return order_crossover(population)
    # partially mapped crossover (PMX)
    elif args.crossover == 2:
        return partially_mapped_crossover(population)
    else:
        raise Exception("selection argument has to be either 1 or 2")

def swap_mutate(population):
    """Uses swap mutation to mutate a whole population, based
    on given probability args.mutation_p

    Args:
        population ([chromosomes])

    Returns:
        population: mutated Population
    """
    mutated_population = []
    for chromosome in population:
        if random.random() < args.mutation_p:
            chromosome = chromosome.route
            # randomly choose 2 genes
            gene_1 = random.randint(0, len(chromosome) - 1)
            gene_2 = random.randint(0, len(chromosome) - 1)
            # swap the 2 genes
            chromosome[gene_1], chromosome[gene_2] = chromosome[gene_2], chromosome[gene_1]
            mutated_population.append(Chromosome(chromosome))
        else:
            # add chromosome without mutation
            mutated_population.append(chromosome)
    return mutated_population

def insertion_mutate(population):
    """Uses insertion mutation to mutate a whole population, based
    on given probability args.mutation_p

    Args:
        population ([chromosomes])

    Returns:
        population: mutated Population
    """
    mutated_population = []
    for chromosome in population:
        if random.random() < args.mutation_p:
            chromosome = chromosome.route
            # randomly choose index for gene
            index_pick = random.randint(0, len(chromosome) - 1)
            index_new = random.randint(0, len(chromosome) - 2)
            # extract gene
            chosen_gene = chromosome.pop(index_pick)
            # insert in new position
            chromosome.insert(index_new, chosen_gene)
            mutated_population.append(Chromosome(chromosome))
        else:
            # add chromosome without mutation
            mutated_population.append(chromosome)
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

def get_shortest_distance(population):
    return sorted([1/chromosome.fitness for chromosome in population])[0]

def main():
    """Creates initial population and then starts the genetic algorithm.
    Solutions will be saved in a file.
    """
    fitness_progress = []
    start_time = datetime.now()

    # get city coordinates from .tsp file, specified as a CLI argument
    get_cities(args.input_file)

    # create a random population
    population = create_population(200, len(CITIES))
    # save fitness of initial population
    fitness_progress.append(get_shortest_distance(population))

    # generations
    for _ in range(0, 2000):
        # randomly select new population influenced on their fitness

        selection = select_chromosomes(population)

        # breed random chromosomes
        breed = crossover(selection)

        # mutate some of the chromosomes
        population = mutate(breed)

        # track fitness progress
        fitness_progress.append(get_shortest_distance(population))

        # end if runtime > 2min 30sec
        if (datetime.now() - start_time).total_seconds() > 150:
            break

    # get best route
    best_route = sorted(population, key=lambda x: x.fitness, reverse=True)[0]
    print(best_route.route)
    print(f"Anzahl Evaluationen: {str(EVALUATIONS)}")

    timestamp = datetime.now().strftime("%H_%M_%S")

    # export csv file with fitness progress
    with open(os.path.join(args.folder, f"data_genetic_algorithm_{timestamp}.csv"), "w") as file_out:
        file_out.write("Generations\tFunctionValue\n")
        for i, fitness in enumerate(fitness_progress):
            file_out.write(f"{str(i)}\t{fitness:.20f}\n")

    print(f"min = {str(get_shortest_distance(population))} | crossover_p = {str(args.crossover_p)} \
        | mutation_p = {str(args.mutation_p)}")

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description='Genetischer Algorithmus in Python \
        für das TSP')

    parser.add_argument('input_file', type=str,
                        help='Specify input file (in .tsp format)')

    # change probabilities
    parser.add_argument('crossover_p', type=float, help='Probability of crossover')
    parser.add_argument('mutation_p', type=float, help='Probability of mutation')

    # change selection, crossover, mutation method
    parser.add_argument('selection', type=int, help="1 for tournament selection, \
        2 for rank based selection")
    parser.add_argument('crossover', type=int, help="1 for order crossover, 2 for \
        partially mapped crossover")
    parser.add_argument('mutation', type=int, help="1 for swapping, 2 for\
        insertion")

    parser.add_argument('--folder', type=str, help='Specify output folder for the csvs',\
        default="")

    args = parser.parse_args()

    main()

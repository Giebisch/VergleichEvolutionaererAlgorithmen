import argparse
import random
import os
from datetime import datetime
import numpy
import math

args = None
EVALUATIONS = 0

# generates random number for the mutation of the strategy parameters
N_0_1 = numpy.random.normal(0, 1)

### Evolution strategy

class Chromosome:
    """Class to save the fitness and strategy parameters of the chromosomes
    """
    def __init__(self, genes, sparams):
        self.genes = genes
        self.sparams = sparams
        self.fitness = get_fitness(self.genes)

def create_population(popsize, length):
    """Creates a new population with the size popsize and
    length-many genes. Sets step sizes to 5% of the search
    space size

    Args:
        popsize: size of population
        length: number of genes per chromosome

    Returns:
        population
    """
    population = []
    for _ in range(0, popsize):
        new_chromosome = [random.uniform(args.min, args.max) for _ in range(length)]
        # set mutation step sizes to 2.5% of search space size
        sparams = [(args.max - args.min) * 0.025 for _ in range(length)]
        population.append(Chromosome(new_chromosome, sparams))
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
    value = genes[0] ** 2 + genes[1] ** 2 + genes[2] ** 2
    return value

def two_point_crossover(population):
    """Uses two point crossover to create a new population

    Args:
        population ([chromosomes])

    Returns:
        new population
    """
    new_population = []
    while len(new_population) < args.lam:
        parent1, parent2 = random.sample(population, k=2)

        # get the two crossover points
        index1 = random.randint(0, len(parent1.genes) - 1)
        index2 = random.randint(0, len(parent2.genes) - 1)

        if index1 > index2:
            index1, index2 = index2, index1

        # create child's genes
        child_genes = parent1.genes[0:index1] + parent2.genes[index1:index2] \
            + parent1.genes[index2:]
        # create child's strategy parameters
        child_sparams = parent1.sparams[0:index1] + parent2.sparams[index1:index2] \
            + parent1.sparams[index2:]

        new_population.append(Chromosome(child_genes, child_sparams))

    return new_population

def uniform_crossover(population):
    """Uses uniform crossover to create a new population

    Args:
        population ([chromosomes])

    Returns:
        new population
    """
    new_population = []
    while len(new_population) < args.lam:
        parent1, parent2 = random.sample(population, k=2)

        # in ES parents only produce one child
        child_genes = []
        child_sparams = []

        for i,_ in enumerate(parent1.genes):
            if random.randint(0, 1) == 0:
                # choose gene and strategy parameter of parent1
                child_genes.append(parent1.genes[i])
                child_sparams.append(parent1.sparams[i])
            else:
                # choose gene and strategy parameter of parent2
                child_genes.append(parent2.genes[i])
                child_sparams.append(parent2.sparams[i])

        new_population.append(Chromosome(child_genes, child_sparams))

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

def mutation(chromosome):
    """Adds perturbation to each gene of [chromosome]
    The scaling is based on the strategy parameters [chromosome.sparams]

    Args:
        [chromosome]

    Returns:
        population: mutated Population
    """
    tau_1 = 0.1
    tau_2 = 0.2
    min_svalue = 0.000001
    # maximum step size: 5% of total search space
    max_svalue = (args.max - args.min) * 0.05

    new_genes = []
    new_sparams = []
    for i, gene in enumerate(chromosome.genes):
        gene = numpy.random.normal(gene, chromosome.sparams[i])
        if gene < args.min: gene = args.min
        if gene > args.max: gene = args.max
        new_genes.append(gene)
        # änderung des sparam
        new_sparam = chromosome.sparams[i] * math.exp(tau_1 * N_0_1 + tau_2 * numpy.random.normal(0, 1))
        if new_sparam < min_svalue: new_sparam = min_svalue
        if new_sparam > max_svalue: new_sparam = max_svalue

        new_sparams.append(new_sparam)
    return Chromosome(new_genes, new_sparams)

def mutate(population):
    new_population = []
    for chromosome in population:
        new_population.append(mutation(chromosome))
    return new_population

def get_best_solution(population):
    return sorted(population, key= lambda x: x.fitness, reverse=True)[0]

def main():
    """Creates initial population and then starts the evolutionary strategy
    Solutions will be saved in a file.
    """
    fitness_progress = []
    start_time = datetime.now()

    # create a random population
    population = create_population(args.mu, 3)
    # save fitness of initial population
    fitness_progress.append(get_function_value(get_best_solution(population).genes))

    # generations
    for _ in range(0, 28):

        # breed chromosomes
        breed = crossover(population)

        # mutate the chromosomes
        mutated_breed = mutate(breed)

        # select new population based on their fitness
        population = select(mutated_breed, population)

        # track fitness progress
        fitness_progress.append(get_function_value(get_best_solution(population).genes))

        # end if runtime > 20 seconds
        if (datetime.now() - start_time).total_seconds() > 20:
            break

    best_solution = get_best_solution(population)
    print(best_solution.genes)
    print(best_solution.sparams)
    print(f"Anzahl Evaluationen: {str(EVALUATIONS)}")

    timestamp = datetime.now().strftime("%H_%M_%S")

    # export csv file with fitness progress
    with open(os.path.join(args.folder, f"data_genetic_algorithm_{timestamp}.csv"), "w") as file_out:
        file_out.write("Generations\tFunctionValue\n")
        for i, fitness in enumerate(fitness_progress):
            file_out.write(f"{str(i)}\t{fitness:.20f}\n")

    print(f"Fitness = {best_solution.fitness:.20f} | mu = {str(args.mu)} \
        | lambda = {str(args.lam)}")

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description='Evolutionsstrategien in Python \
        für das Optimieren einer Funktion')

    # change probabilities
    parser.add_argument('mu', type=int, help='Set µ (mu)')
    parser.add_argument('lam', type=int, help='Set λ (lambda)')

    # change selection, crossover, mutation method
    parser.add_argument('selection', type=int, help="1 for (mu,lambda), \
        2 for (mu+lambda)")
    parser.add_argument('crossover', type=int, help="1 for two point crossover, 2 for \
        uniform crossover")

    # upper and lower bound
    parser.add_argument('min', type=float, help="Set lower bound for search space")
    parser.add_argument('max', type=float, help="Set upper bound for search space")

    parser.add_argument('--folder', type=str, help='Specify output folder for the csvs',\
        default="")

    args = parser.parse_args()

    main()

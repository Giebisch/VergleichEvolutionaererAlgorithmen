import argparse
import math
import random
from datetime import datetime
from copy import deepcopy

CITIES = []
args = None
EVALUATIONS = 0

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
### Hill climbing

def create_init_tour(popsize):
    population = list(range(1, popsize + 1))
    random.shuffle(population)
    return population

def get_fitness(tour):
    global EVALUATIONS

    def get_distance(__city1, __city2):
        # returns euclidean distance
        return math.sqrt( (__city2[0] - __city1[0])**2 + (__city2[1] - __city1[1])**2 )

    distance = 0
    for i in range(0, len(tour) - 1):
        city1 = CITIES[tour[i] - 1]
        city2 = CITIES[tour[i+1] - 1]
        distance += get_distance(city1, city2)
    # add distance from last to first city (round trip)
    distance += get_distance(CITIES[tour[-1] - 1], CITIES[tour[0] - 1])
    
    EVALUATIONS += 1
    return 1/distance

def swapped_cities(tour):
    new_tours = []

    for _ in range(219):
        index1 = random.randint(1, len(tour) - 2)
        index2 = random.randint(1, len(tour) - 2)
        copy = deepcopy(tour)
        copy[index1], copy[index2] = copy[index2], copy[index1]
        new_tours.append(copy)

    return new_tours

def main():
    fitness_progress = []
    start_time = datetime.now()

    # get city coordinates from .tsp file, specified as a CLI argument
    get_cities(args.input_file)

    # create a random population
    tour = create_init_tour(len(CITIES))

    # generations
    for i in range(0, 2000):
        best_tour = sorted(swapped_cities(tour), key= lambda x: get_fitness(x), reverse=True)[0]

        if get_fitness(best_tour) > get_fitness(tour):
            tour = best_tour

        # track distance progress
        fitness_progress.append(1/get_fitness(tour))
        
        # end if runtime > 120 seconds
        if (datetime.now() - start_time).total_seconds() > 120:
            break

    # plot best route
    timestamp = datetime.now().strftime("%H_%M_%S")

    print(tour)
    print(f"Anzahl Evaluationen: {str(EVALUATIONS)}")

    timestamp = datetime.now().strftime("%H_%M_%S")

    # export csv file with fitness progress
    with open(f"data_{timestamp}.csv", "w") as file_out:
        file_out.write("Generations\tFunctionValue\n")
        for i, fitness in enumerate(fitness_progress):
            file_out.write(f"{str(i)}\t{fitness:.20f}\n")

    print(f"min = {str(1/get_fitness(tour))}")
    
if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description='Hill climbing in Python für das TSP')

    parser.add_argument('input_file', type=str,
                        help='Specify input file (in .tsp format)')

    args = parser.parse_args()

    main()

Dieser Code ist Bestandteil der Bachelorarbeit "Analyse und Vergleich ausgewählter evolutionärer Algorithmen" von Rafael Giebisch, 2021.

# Genetische Algorithmen für TSP

## Inhalt

Dieser Code optimiert die in der Bachelorarbeit vorgestellte Aufgabe eil101.tsp mittels genetischer Algorithmen. Mithilfe der eingegebenen Parameter können Selektions-, Rekombinations- und Mutationsoperator ausgewählt werden.

## Verwendung

Zur Ausführung des Codes wird mindestens Python 3.8.5 benötigt.

Für Hinweise zum Verwenden des Codes `python genetic_algorithm.py --help`  ausführen. Dann erscheint folgender Hinweistext:

    usage: genetic_algorithm.py [-h] [--folder FOLDER] input_file crossover_p mutation_p selection crossover mutation

    Genetischer Algorithmus in Python für das TSP

    positional arguments:
    input_file       Specify input file (in .tsp format)
    crossover_p      Probability of crossover
    mutation_p       Probability of mutation
    selection        1 for tournament selection, 2 for rank based selection
    crossover        1 for order crossover, 2 for partially mapped crossover
    mutation         1 for swapping, 2 for insertion

    optional arguments:
    -h, --help       show this help message and exit
    --folder FOLDER  Specify output folder for the csvs

    optional arguments:
    -h, --help       show this help message and exit
    --folder FOLDER  Specify output folder for the csvs

Example usage:

    python genetic_algorithm.py eil101.tsp 0.8 0.3 1 1 1
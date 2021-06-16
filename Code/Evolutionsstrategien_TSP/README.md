Dieser Code ist Bestandteil der Bachelorarbeit "Analyse und Vergleich ausgewählter evolutionärer Algorithmen" von Rafael Giebisch, 2021.

# Evolutionsstrategien für TSP

## Inhalt

Dieser Code optimiert die in der Bachelorarbeit vorgestellte Aufgabe eil101.tsp mittels Evolutionsstrategien. Mithilfe der eingegebenen Parameter können Selektions- und Rekombinationsoperatoren ausgewählt werden.

## Verwendung

Zur Ausführung des Codes wird mindestens Python 3.8.5 benötigt. Außerdem wird numpy vorausgesetzt.

Für Hinweise zum Verwenden des Codes `python evolutionsstrategien.py --help`  ausführen. Dann erscheint folgender Hinweistext:

    usage: evolutionsstrategien.py [-h] [--folder FOLDER] input_file mu lam selection crossover mutation

    Evolutionsstrategien in Python für das TSP

    positional arguments:
    input_file       Specify input file (in .tsp format)
    mu               Set µ (mu)
    lam              Set λ (lambda)
    selection        1 for (mu,lambda), 2 for (mu+lambda)
    crossover        1 for order crossover, 2 for partially mapped crossover
    mutation         1 for swap mutate, 2 for insertion_mutate

    optional arguments:
    -h, --help       show this help message and exit
    --folder FOLDER  Specify output folder for the csvs

Example usage:

    python evolutionsstrategien.py eil101.tsp 15 105 1 2 1
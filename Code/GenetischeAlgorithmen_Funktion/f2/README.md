Dieser Code ist Bestandteil der Bachelorarbeit "Analyse und Vergleich ausgewählter evolutionärer Algorithmen" von Rafael Giebisch, 2021.

# Genetische Algorithmen für Funktion f_2

## Inhalt

Dieser Code optimiert die in der Bachelorarbeit vorgestellte Funktion f_2 mittels genetischer Algorithmen. Mithilfe der eingegebenen Parameter können Selektions-, Rekombinations- und Mutationsoperator ausgewählt werden.

## Verwendung

Zur Ausführung des Codes wird mindestens Python 3.8.5 benötigt. Außerdem wird numpy vorausgesetzt.

Für Hinweise zum Verwenden des Codes `python genetic_algorithm.py --help`  ausführen. Dann erscheint folgender Hinweistext:

    usage: genetic_algorithm.py [-h] [--folder FOLDER] crossover_p mutation_p selection crossover mutation min max

    Genetischer Algorithmus in Python für das Optimieren einer Funktion

    positional arguments:
    crossover_p      Probability of crossover
    mutation_p       Probability of mutation
    selection        1 for tournament selection, 2 for rank based selection
    crossover        1 for two point crossover, 2 for uniform crossover
    mutation         1 for uniform mutation, 2 for gaussian mutation
    min              Set lower bound for search space
    max              Set upper bound for search space

    optional arguments:
    -h, --help       show this help message and exit
    --folder FOLDER  Specify output folder for the csvs

Example usage:

    python genetic_algorithm.py 0.9 0.1 1 2 1 -2.048 2.048
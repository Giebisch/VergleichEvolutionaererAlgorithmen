Dieser Code ist Bestandteil der Bachelorarbeit "Analyse und Vergleich ausgewählter evolutionärer Algorithmen" von Rafael Giebisch, 2021.

# Evolutionsstrategien für Funktion f_2

## Inhalt

Dieser Code optimiert die in der Bachelorarbeit vorgestellte Funktion f_2 mittels Evolutionsstrategien. Mithilfe der eingegebenen Parameter können Selektions- und Rekombinationsoperatoren ausgewählt werden.

## Verwendung

Zur Ausführung des Codes wird mindestens Python 3.8.5 benötigt. Außerdem wird numpy vorausgesetzt.

Für Hinweise zum Verwenden des Codes `python evolutionsstrategien.py --help`  ausführen. Dann erscheint folgender Hinweistext:

    usage: evolutionsstrategien.py [-h] [--folder FOLDER] mu lam selection crossover min max

    Evolutionsstrategien in Python für das Optimieren einer Funktion

    positional arguments:
    mu               Set µ (mu)
    lam              Set λ (lambda)
    selection        1 for (mu,lambda), 2 for (mu+lambda)
    crossover        1 for two point crossover, 2 for uniform crossover
    min              Set lower bound for search space
    max              Set upper bound for search space

    optional arguments:
    -h, --help       show this help message and exit
    --folder FOLDER  Specify output folder for the csvs

Example usage:

    python evolutionsstrategien.py 15 105 1 1 -2.048 2.048 
Dieser Code ist Bestandteil der Bachelorarbeit "Analyse und Vergleich ausgewählter evolutionärer Algorithmen" von Rafael Giebisch, 2021.

# Hill climbing für Funktion f_1

## Inhalt

Dieser Code optimiert die in der Bachelorarbeit vorgestellte Funktion f_1 mittels Hill climbings.

## Verwendung

Zur Ausführung des Codes wird mindestens Python 3.8.5 benötigt. Außerdem wird numpy vorausgesetzt.

Für Hinweise zum Verwenden des Codes `python hillclimb.py --help`  ausführen. Dann erscheint folgender Hinweistext:

    usage: hillclimbing.py [-h] [--folder FOLDER] min max

    Hill climbing in Python für das Optimieren einer Funktion

    positional arguments:
    min              Set lower bound for search space
    max              Set upper bound for search space

    optional arguments:
    -h, --help       show this help message and exit
    --folder FOLDER  Specify output folder for the csvs

Example usage:

    hillclimbing.py -5.12 5.12
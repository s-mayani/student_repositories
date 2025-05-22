#!/bin/bash

# Finds `BeamStatistics.csv` File in directory given by commandline argument, \
# removes parenthesis and stores it into a file called `Moments.csv`
find $1 -type f -name "BeamStatistics.csv" -execdir sh -c 'cat "$1" | tr -d "()" > Moments.csv' sh {} \;

#!/usr/bin/gnuplot

# Set terminal and output
set terminal pdf size 3,3 font "CMU serif"
set output "trajectory_radiating.pdf"

# Set font

# Set grid
set grid

# Set log scale for both x and y axes

#set yrange [0.1:200]
# Define the path variable
path_variable = "/run/media/manuel/docpart/gitclones/ippl_seeding_bugfix/ippl_oo/build/"

# Set plot title
#set title "Particle trajectory in constant external B-Field"

# Set axis labels
set xlabel "X [pl]"
set ylabel "Y [pl]"

# Plot the data from the file and a function
plot path_variable.'/ppos.txt' u 1:2 w lines lc "red" title "Particle"

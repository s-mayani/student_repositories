#! /usr/bin/gnuplot
set title "Total radiation of particle in circular orbit, beta=0.9, gamma=2.29"
set terminal pdf size 4,3 font "CMU serif"
set xlabel "Time [PlanckT]"
set ylabel "Power [PlanckP]"
set output "cradplot.pdf"
set key box
set key bottom center
set grid lw 2
path_variable = "/run/media/manuel/docpart/gitclones/ippl_seeding_bugfix/ippl_oo/build/"
#plot path_variable.'boundary_radation.txt' w lines lw 2 title "Radiation over sphere", \
#path_variable.'boundary_radation.txt' u 1:($2 - $2 + 24.1) w lines title "Larmor",
plot path_variable.'cumulative_radiation.txt' w lines lw 2 title "Cumulative Radiation over sphere", \
path_variable.'cumulative_radiation.txt' u 1:($1 * ($2 - $2 + 24.1)) w lines title "Cumulative Larmor",
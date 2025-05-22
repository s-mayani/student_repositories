#! /usr/bin/gnuplot
set terminal svg size 800,600 font "CMU Serif, 25"
set output "weak_scaling.svg"
set logscale xy 2
set grid lw 2
set key spacing 1.2 box
set yrange[1:4]
set xlabel "Number of GPU Ranks"
set ylabel "Runtime [min.]"
plot 'weak_scaling.txt' u 1:($2 * 10 / $1 / 60) w points title "Real scaling" ps 2, 'weak_scaling.txt' u 1:($1 - $1 + 97.1 / 60) w lines lw 3 title "Ideal scaling"
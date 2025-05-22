#! /usr/bin/gnuplot
set terminal svg size 800,600 font "CMU Serif, 25"
set output "strong_scaling.svg"
set logscale xy 2
set grid lw 2
set yrange [4:]
set key spacing 1.2 box
set xlabel "Number of GPU Ranks"
set ylabel "Runtime [min.]"
plot 'strong_scaling.txt' u 1:(10 * $2 / 60) w points title "Real scaling" ps 2, 'strong_scaling.txt' u 1:(2380 / ($1 * 60)) w lines lw 3 title "Ideal scaling"
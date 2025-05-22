#! /usr/bin/gnuplot

set logscale xy 2
set xtics 2
set ytics 2
set xlabel "3D Grid extent [N]"
set ylabel "Runtime [ms]"
set grid lw 2
set terminal svg size 800,600 font "CMU Serif, 20"
set output "stencils_vs_fft_merlin.svg"
set xrange [64:]
set key top left box
plot 'cufft_times.txt' u 1:($2/40) w lines lw 3 title "cuFFT 5-point", 'naive3_times.txt' u 1:($2/40) w lines lw 3 title "Naive 3-point Stencil", 'naive5_times.txt' u 1:($2/40) w lines lw 3 title "Naive 5-point Stencil", 'naive7_times.txt' u 1:($2/40) w lines lw 3 title "Naive 7-point Stencil"

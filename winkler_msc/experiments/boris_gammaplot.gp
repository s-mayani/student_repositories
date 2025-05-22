#! /usr/bin/gnuplot
set terminal svg font "CMU Serif, 20" size 1200, 600
set output "boris_gamma.svg"
set multiplot layout 1,2
set logscale xy 2
set grid lw 2
set xtics (0.5,1,2,3,4,5,6,7,8)
set ytics (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
set xrange [0.25:8.1]
set xlabel "Undulator parameter K [unitless]"
set ylabel "Ratio of γ_0  to γ"
set key box bottom left font ", 20"
plot 'boris_gammas.txt' u 1:($2 / 20) w lines lw 2 lc "blue" title "Measured", \
     'boris_gammas.txt' u 1:(1.0 / sqrt(1 + $1 * $1 * 0.5)) w lines lw 2 lc "red" title "Approx"
plot 'boris_gammas.txt' u 1:($2 / 20) w lines lw 2 lc "blue"  title "Measured", \
     'boris_gammas.txt' u 1:(1.0 / sqrt(1 + $1 * $1 * 0.51 + $1 * $1 * $1 * 0.00375)) w lines lw 2 lc "green" title "Better Approx"

#plot 'boris_gammas.txt' u 1:($2 / 20) w lines lw 2 title "γ=20 (Measured)",\
#     'boris_gammas.txt' u 1:(1.0 / sqrt(1 + $1 * $1 * 0.5)) w lines lw 2 title "γ=20 (Approx)",\
#     'boris_gammas.txt' u 1:($3 / 100) w lines lw 2 title "γ=100 (Measured)", \
#     'boris_gammas.txt' u 1:(1.0 / sqrt(1 + $1 * $1 * 0.5)) w lines lw 2 title "γ=100 (Approx)",\
#     'boris_gammas.txt' u 1:($4 / 1000) w lines lw 2 title "γ=1000 (Measured)", \
#     'boris_gammas.txt' u 1:(1.0 / sqrt(1 + $1 * $1 * 0.5)) w lines lw 2 title "γ=1000 (Approx)"\
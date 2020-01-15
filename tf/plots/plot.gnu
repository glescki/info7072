set title "Comparison between CPU and GPU implementation of Needleman-Wunsch" font ",14"
# set label "N=[124346, 175852, 248692, 351704], NB=192, CPU freq=2.2, \nPxQ=[2x2, 2x4, 4x4, 4x8]" center at graph 0.5, char 2 font ",12"
# set bmargin 6

set ylabel "Time in ns"
set xlabel "Sequence length"
set y2label "Speedup"
#set grid

# set logscale y
set xtics 0,10000,500
# set xtics rotate by -45

# color definitions
set style line 1 lc rgb '#e41a1c' pt 1 ps 1 lt 1 lw 2 # --- red
set style line 2 lc rgb '#4daf4a' pt 1 ps 1.5 lt 1 lw 2 # --- green
set style line 3 lc rgb '#377eb8' pt 7 ps 1.5 lt 1 lw 2 # --- blue

set offset -0.6,-0.6,0,0

set style data histogram
set style histogram clustered gap 1
# set style fill solid border -1
set boxwidth 0.75
set xtic scale 0
set style fill solid
set ytics nomirror
set y2tics

set key opaque top left

# set terminal png
# set output 'fig.png'

plot 'gpu.dat' u ($2 / 1000): xtic(1) w histogram ls 2 t "GPU" axis x1y1, \
     'cpu.dat' u ($2 / 1000): xtic(1) w histogram ls 3 t "CPU" axis x1y1, \
     'speedup.dat' u 2 w linespoint ls 1 t "Speedup" axis x1y2

pause -1

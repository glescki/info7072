set title "Comparison between blur kernels (GTX750ti)" font ",14"
# set label "N=[124346, 175852, 248692, 351704], NB=192, CPU freq=2.2, \nPxQ=[2x2, 2x4, 4x4, 4x8]" center at graph 0.5, char 2 font ",12"
# set bmargin 6

set ylabel "Time in μs"
set xlabel "Images (10=sunflower, 11=Lena)"
set y2label "Speedup"
#set grid

# set logscale y
#set ytics 0,500,30000
#set xrange[0:11]
#set xtics 0,1,11
#set xtics rotate by -45

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


plot 'ImageBlur.dat' using ($2 * 1000000): xtic(1) ls 1 with histogram title "ImageBlur (no shared memory)", \
     'ImageBlurSHM.dat' using ($2 * 1000000): xtic(1) ls 2 with histogram title "ImageBlurSHM only", \
     'ImageBlurSHMC.dat' using ($2 * 1000000): xtic(1) ls 3 with histogram title "ImageBlurSHM + layout transformation", \
     'SpdupSHM.dat' axis x1y2 with linespoint ls 2 title "speedup ImageBlurSHM only", \
     'SpdupSHMC.dat' axis x1y2 with linespoint ls 3 title "speedup ImageBlurSHM + layout transformation"

pause -1

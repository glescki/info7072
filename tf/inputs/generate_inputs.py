import sys
import numpy as np
from numpy.random import choice, randint

n = range(500, 10001, 500)

nt = ['A', 'T', 'C', 'G']
pr = [0.3, 0.3, 0.2, 0.2]

for i in n:
    ref = choice(nt, i, p=pr)
    ref = np.append(ref, "\n")

    output = "input_"+str(i)+".dat"

    output_file = open(output, 'w')

    np.savetxt(output_file, ref, newline="", fmt='%s')

    output_file.close()

    output_file = open(output, 'a')

    align = choice(nt, i, p=pr)
    align = np.append(align, "\n")

    np.savetxt(output_file, align, newline="", fmt='%s')

    m = randint(i * 0.1, i + 1, size=4)

    for j in m:
        align = choice(nt, j, p=pr)
        align = np.append(align, "\n")

        np.savetxt(output_file, align, newline="", fmt='%s')

    output_file.close()

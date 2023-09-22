'''
Thanks to Rishal Aggarwal for original function (https://github.com/devalab/DeepPocket/blob/main/get_centers.py)
'''

import os
import re
import numpy as np

def get_centers(directory):
    barycenters_file = open(f"{directory}/bary_centers.txt", 'w')
    for filename in os.listdir(directory):
        centers = []
        masses = []
        if filename.endswith('vert.pqr'):
            residue_number = int(re.search(r'\d+', filename).group())
            with open(f"{directory}/{filename}") as file:
                for line in file:
                    if line.startswith('ATOM'):
                        center = list(map(float, re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", ' '.join(line.split()[5:]))))[:3]
                        mass = float(line.split()[-1])
                        centers.append(center)
                        masses.append(mass)
            centers = np.asarray(centers)
            masses = np.asarray(masses)
            xyzm = (centers.T * masses).T
            xyzm_sum = xyzm.sum(axis=0)
            center_of_mass = xyzm_sum / masses.sum()
            barycenters_file.write(f"{residue_number}\t{center_of_mass[0]}\t{center_of_mass[1]}\t{center_of_mass[2]}\n")

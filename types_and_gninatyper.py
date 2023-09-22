'''
Thanks to Rishal Aggarwal for original function (https://github.com/devalab/DeepPocket/blob/main/types_and_gninatyper.py)
'''
import molgrid
import struct
import numpy as np
import os,pathlib
import sys


def create_gninatype_file(pdb_file):
    # Creates gninatype file for model input
    gninatype_file = pdb_file.replace('.pdb', '.gninatypes')

    with open(pdb_file.replace('.pdb', '.types'), 'w') as file:
        file.write(pdb_file)

    atom_map = molgrid.FileMappedGninaTyper(f'{pathlib.Path(os.path.realpath(__file__)).resolve().parent}/features')
    dataloader = molgrid.ExampleProvider(atom_map, shuffle=False, default_batch_size=1)

    train_types = pdb_file.replace('.pdb', '.types')
    dataloader.populate(train_types)

    example = dataloader.next()
    coords = example.coord_sets[0].coords.tonumpy()
    types = example.coord_sets[0].type_index.tonumpy()
    types = np.int_(types)

    with open(gninatype_file, 'wb') as file:
        for i in range(coords.shape[0]):
            file.write(struct.pack('fffi', coords[i][0], coords[i][1], coords[i][2], types[i]))

    os.remove(train_types)

    return gninatype_file

def create_types_file(input_file, protein_name):
    # Creates types file for model predictions
    output_file = input_file.replace('.txt', '.types')
    output = open(output_file, 'w')
    with open(input_file, 'r') as input:
        for line in input:
            output.write(' '.join(line.split()) + ' ' + protein_name + '\n')
    return output_file


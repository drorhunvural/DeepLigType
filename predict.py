import torch
import os
import molgrid
import argparse
import gc
import pathlib
import time
import torch.nn as nn
import torch.nn.functional as F
import csv
import shutil

from cleanpdb import pdb_clean
from get_centers import get_centers
from types_and_gninatyper import create_gninatype_file,create_types_file

from deeplearningmodels.cbam import ResNet18_CBAM_3D, BasicBlock3D, ChannelAttention3D, SpatialAttention3D

current_directory = os.path.dirname(os.path.abspath(__file__))
bestmodels_dir = os.path.join(current_directory, 'bestmodels')

def parse_arguments(args=None):

    parser = argparse.ArgumentParser(description='Classify by Ligand Type')

    parser.add_argument('-p', '--protein', type=str, required=False, help="Input PDB file")
    parser.add_argument('-t', '--trainedpth', type=str, required=False, help="Trained Model")

    args = parser.parse_args(args)

    arg_dict = vars(args)
    arg_str = ''
    for name, value in arg_dict.items():
        if value != parser.get_default(name):
            arg_str += f' --{name}={value}'

    return args, arg_str

def to_cuda(*models):
    return [model.to("cuda") for model in models]

if __name__ == '__main__':
    (args, cmdline) = parse_arguments()
    
    project_path = os.path.dirname(os.path.abspath(__file__))
    trainedpth = args.trainedpth

    trainedpth_dir = os.path.join(bestmodels_dir, trainedpth)

    deep_model = torch.load(trainedpth_dir)

    protein_file= args.protein
    pro_id = protein_file.split("/")[-1].split(".")[0] # take four digit protein id 
    protein_nowat_file=protein_file.replace('.pdb','_nowat.pdb')

    pdb_clean(protein_file,protein_nowat_file) #clean pdb file and remove hetero atoms/non standard residues
    
    os.system('fpocket -f '+ protein_nowat_file) # fpocket

    fpocket_dir=os.path.join(protein_nowat_file.replace('.pdb','_out'),'pockets')

    fpocket_result_folder = pathlib.Path(fpocket_dir).parent #xxx_nowat_out folder

    get_centers(fpocket_dir) #create bary_centers.txt

    barycenter_file=os.path.join(fpocket_dir,'bary_centers.txt')

    protein_gninatype=create_gninatype_file(protein_nowat_file) # dir of gninatype
    
    class_types=create_types_file(barycenter_file,protein_gninatype) # create bary_centers_ranked.types

    types_lines=open(class_types,'r').readlines()

    batch_size = len(types_lines)
    #avoid cuda out of memory
    if batch_size>50:
        batch_size=50

    gmaker2 = molgrid.GridMaker(binary=False)
    dims = gmaker2.grid_dimensions(24) 
    tensor_shape_2 = (1,)+dims #(1, 24, 48, 48, 48)

    inputfile_name = 'inputfile.types'
    inputfile_dir = os.path.join(project_path, inputfile_name)
    #inputfile_dir = '/content/inputfile.types' # inputfile.types contains X,Y,Z,xxxx_nowat.gninatypes for exampleprovider


    formatted_lines = []

    # Read the input file
    with open(os.path.abspath(class_types), "r") as infile:
        for line in infile:
            parts = line.strip().split()  # Split the line into parts
            if len(parts) == 5:
                # Extract the filename from the last column
                filename_parts = parts[4].split('/')
                filename = filename_parts[-1]
                
                # Format the line with extracted values
                formatted_line = f"{parts[1]} {parts[2]} {parts[3]} {filename}"
                formatted_lines.append(formatted_line)

    # Write the formatted lines to the output file
    with open(inputfile_dir, "w") as outfile:
        for line in formatted_lines:
            outfile.write(line + "\n")
    

    e_test_1 = molgrid.ExampleProvider(data_root= project_path, shuffle=False,stratify_receptor=True, balanced = False)
    e_test_1.populate(inputfile_dir)

    input_tensor_1 = torch.zeros(tensor_shape_2, dtype=torch.float32, device='cuda') #[1, 24, 48, 48, 48]
    float_labels_1 = torch.zeros((1,3), dtype=torch.float32, device='cuda')
    
    categorization_prediction = []
    for i in range(batch_size):        
            batch =e_test_1.next_batch()
            # extract centers of batch datapoints
            batch.extract_labels(float_labels_1)
            centers = float_labels_1[:,0:]

            for b in range(1):
              center = molgrid.float3(float(centers[b][0]),float(centers[b][1]),float(centers[b][2]))
                    # Update input tensor with b'th datapoint of the batc
              gmaker2.forward(center,batch[b].coord_sets[0],input_tensor_1[b])
           
            output = deep_model(input_tensor_1[:,:24])
      
            predicted = torch.argmax(output,dim=1)
            categorization_prediction.append(predicted.item())

    mapping = {0: 'Other', 1: 'Antagonist', 2: 'Inhibitor', 3: 'Activator', 4: 'Agonist'}

    def get_category(prediction):
      if prediction == 0:
          return "Other"
      elif prediction == 1:
          return "Antagonist"
      elif prediction == 2:
          return "Inhibitor"
      elif prediction == 3:
          return "Activator"
      elif prediction == 4:
          return "Agonist"
      else:
          return "None" # Return a None string for any other values

    with open(inputfile_dir, 'r') as input_file:
     lines = input_file.readlines()

    inputfiledata = [line.strip().split() for line in lines]

    for i, row in enumerate(inputfiledata):
        prediction = categorization_prediction[i]
        row.append(get_category(prediction))
        protein_id = row[-2]  # Access the second-to-last column
        protein_id_parts = protein_id.split('_')
        if len(protein_id_parts) > 0:
            row[-2] = protein_id_parts[0]
  
    csvfilename = "{}.csv".format(pro_id)

    with open(csvfilename, 'w', newline='') as output_csv_file:
      csv_writer = csv.writer(output_csv_file)
      csv_writer.writerow(["x", "y", "z", "protein_id", "LigandType"])
      csv_writer.writerows(inputfiledata)


    lines.clear()
    shutil.rmtree(fpocket_result_folder)
    os.remove(protein_nowat_file)
    os.remove(protein_gninatype)
    os.remove(inputfile_dir)
  






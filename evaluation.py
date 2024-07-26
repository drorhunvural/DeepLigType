import torch
import os 
import sys
import argparse
import molgrid
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np

from deeplearningmodels.seresnet import SEResNet, ResidualBlock
from deeplearningmodels.cbam import ResNet18_CBAM_3D, BasicBlock3D, ChannelAttention3D, SpatialAttention3D
from deeplearningmodels.cnn import CNNModel
from deeplearningmodels.resnet18 import ResidualBlock_Resnet18, ResNet18
from deeplearningmodels.densenet import DenseNet3D

molgrid.set_random_seed(42)
torch.manual_seed(42)
np.random.seed(42)
num_features = 24
def parse_arguments(args=None):

    parser = argparse.ArgumentParser(description='Classify by Ligand Type')

    parser.add_argument('-t', '--trainedpth', type=str, required=False, help="Trained Model")

    args = parser.parse_args(args)

    arg_dict = vars(args)
    arg_str = ''
    for name, value in arg_dict.items():
        if value != parser.get_default(name):
            arg_str += f' --{name}={value}'

    return args, arg_str

current_directory = os.path.dirname(os.path.abspath(__file__))
bestmodels_dir = os.path.join(current_directory, 'bestmodels')
molcache_path = os.path.join(current_directory, 'dataset', 'pdb.molcache')

test_path = os.path.join(current_directory, 'dataset', 'testfinalv0.types')

if __name__ == '__main__':
    (args, cmdline) = parse_arguments()

    trainedpth = args.trainedpth

    trainedpth_dir = os.path.join(bestmodels_dir, trainedpth)
   
    loaded_model1 =torch.load(trainedpth_dir)
    batch_size = 64
    
    e_test = molgrid.ExampleProvider(stratify_min = 0, stratify_max = 5, stratify_step=1,  shuffle=True, recmolcache = molcache_path,stratify_receptor=True, balanced = False)
    e_test.populate(test_path)
    e_test.size()

    gmaker = molgrid.GridMaker(binary=False)
    dims = gmaker.grid_dimensions(e_test.num_types())
    tensor_shape = (batch_size,)+dims

    # initializations
    test_losses=[]
    test_accuracies = []
    f1_scores= []
    recall_scores= []
    precision_scores= []

    num_iterations = 20
    criterion = nn.CrossEntropyLoss()

    # initializations
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda', requires_grad=True)
    float_labels = torch.zeros((batch_size,4), dtype=torch.float32, device='cuda')

    for epoch in range(num_iterations):
        test_loss = 0
        test_accuracy = 0

        all_labels = []
        all_predicted = []

        with torch.no_grad():
            loaded_model1.eval()
            for i in range(num_iterations):
                batch =e_test.next_batch(batch_size )
                batch.extract_labels(float_labels)
                centers = float_labels[:,1:]
                labels = float_labels[:,0].long().to('cuda')

                for b in range(batch_size):
                    center = molgrid.float3(float(centers[b][0]),float(centers[b][1]),float(centers[b][2]))
                    gmaker.forward(center,batch[b].coord_sets[0],input_tensor[b])

                output = loaded_model1(input_tensor[:,:num_features])
                loss = criterion(output,labels)
                predicted = torch.argmax(output,dim=1)

                accuracy = labels.eq(predicted).sum().float() / batch_size

                # Log loss and accuracy
                test_loss += loss.item()
                test_accuracy += accuracy.item()
                print('Epoch: {} Iteration: {} Loss: {:.4f} Accuracy: {:.2f}%'.format(
                epoch + 1, i + 1, loss.item(), 100. * accuracy.item()))

                # Collect all labels and predictions for the epoch
                all_labels.extend(labels.tolist())
                all_predicted.extend(predicted.tolist())

            # Convert to numpy arrays
            all_labels = np.array(all_labels)
            all_predicted = np.array(all_predicted)

            # Calculate and append metrics
            f1_scores.append(f1_score(all_labels, all_predicted, average='macro'))
            recall_scores.append(recall_score(all_labels, all_predicted, average='macro'))
            precision_scores.append(precision_score(all_labels, all_predicted, average='macro'))

            test_losses.append(test_loss / num_iterations)
            test_accuracies.append(test_accuracy / num_iterations)

            print('Epoch: {} Loss: {:.4f} Accuracy: {:.2f}%'.format(
                  epoch + 1, test_loss / num_iterations, 100. * test_accuracy / num_iterations))

            print("\n Classification Report:")
            print(classification_report(all_labels, all_predicted))

    print('Best Accuracy: {:.3f}%'.format(max(test_accuracies) * 100))
    print('Best F1 Score: {:.3f}'.format(max(f1_scores)))
    print('Best Precision: {:.3f}'.format(max(precision_scores)))
    print('Best Recall: {:.3f}'.format(max(recall_scores)))
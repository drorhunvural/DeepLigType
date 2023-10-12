import molgrid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.nn import init
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,classification_report,auc
from sklearn.metrics import recall_score,precision_score,precision_recall_fscore_support,roc_auc_score,roc_curve
import numpy as np
import os 
import sys
import argparse
from datetime import datetime
from tqdm import tqdm


from deeplearningmodels.seresnet import SEResNet, ResidualBlock
from deeplearningmodels.cbam import ResNet18_CBAM_3D, BasicBlock3D
from deeplearningmodels.cnn import CNNModel
from deeplearningmodels.resnet18 import ResidualBlock_Resnet18, ResNet18
from deeplearningmodels.densenet import DenseNet3D

molgrid.set_random_seed(42)
torch.manual_seed(42)
np.random.seed(42)

current_date = datetime.now().strftime('%Y%m%d') 
current_directory = os.path.dirname(os.path.abspath(__file__))
bestmodel_folder_path = os.path.join(current_directory, "bestmodels")

#base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
train_path = os.path.join(current_directory, 'dataset', 'trainfinalv0.types')
test_path = os.path.join(current_directory, 'dataset', 'testfinalv0.types')
validate_path = os.path.join(current_directory, 'dataset', 'validatefinalv0.types')
molcache_path = os.path.join(current_directory, 'dataset', 'pdb.molcache')

def parse_arguments(args=None):

    parser = argparse.ArgumentParser(description='Classify by Ligand Type')

    parser.add_argument('-m', '--model', type=str, required=False, help="Input PDB file")
  
    args = parser.parse_args(args)

    arg_dict = vars(args)
    arg_str = ''
    for name, value in arg_dict.items():
        if value != parser.get_default(name):
            arg_str += f' --{name}={value}'

    return args, arg_str

def weights_init(m): 
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)

def to_cuda(*models):
    return [model.to("cuda") for model in models]

batch_size = 64

e_train = molgrid.ExampleProvider(stratify_min = 0, stratify_max = 5, stratify_step=1,  shuffle=False, recmolcache = molcache_path,stratify_receptor=False, balanced = False)
e_train.populate(train_path)
e_train.size()

e_validate = molgrid.ExampleProvider(stratify_min = 0, stratify_max = 5, stratify_step=1,  shuffle=False, recmolcache = molcache_path,stratify_receptor=False, balanced = False)
e_validate.populate(validate_path)
e_validate.size()

gmaker = molgrid.GridMaker(binary=False)
dims = gmaker.grid_dimensions(e_train.num_types())
tensor_shape = (batch_size,)+dims

input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda',requires_grad=True)
float_labels = torch.zeros((batch_size,4), dtype=torch.float32, device='cuda')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_tensor, float_labels = input_tensor.to(device), float_labels.to(device)

model_seresnet, model_cbam, model_cnn, model_resnet18, model_densenet = to_cuda(
    SEResNet(ResidualBlock, [2, 2, 2, 2]),
    ResNet18_CBAM_3D(BasicBlock3D, [2, 2, 2, 2]),
    CNNModel(num_classes=5),
    ResNet18(ResidualBlock_Resnet18, [2, 2, 2, 2]),
    DenseNet3D(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=5)
)

if __name__ == '__main__':
    (args, cmdline) = parse_arguments()

    model= args.model
        # Select model based on args.model
    if args.model == "cbam":
        model = model_cbam
    elif args.model == "seresnet":
        model = model_seresnet
    elif args.model == "cnn":
        model = model_cnn
    elif args.model == "resnet18":
        model = model_resnet18
    elif args.model == "densenet":
        model = model_densenet
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model.apply(weights_init)

    optimizerAdam = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    # 24 different properties of proteins

    from datetime import datetime
    validation_loss_min = np.Inf
    validation_acc_max = 0
    best_train_acc_max = 0
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    validation_losses = []
    validation_accuracies = []
    num_epochs = 25
    num_iterations = 100
    validation_num_iterations = 100

    best_train_model_filename = None
    validation_loss_min = np.Inf
    patience = 5  # Number of epochs to wait before early stopping
    counter = 0  # Counter for patience

    for epoch in range(num_epochs):
        for mode in ['train', 'validation']:
            if mode == 'train':
                if best_train_model_filename is not None and epoch > 15:
                    model = torch.load(best_train_model_filename)
                    print('Loading the best training model...')
                model.train()
                train_loss = 0
                train_accuracy = 0

                for i in tqdm(range(num_iterations)):
                        batch = e_train.next_batch(batch_size)
                        batch.extract_labels(float_labels)
                        centers = float_labels[:, 1:]
                        labels = float_labels[:, 0].long().to('cuda')

                        for b in range(batch_size):
                          center = molgrid.float3(float(centers[b][0]),float(centers[b][1]),float(centers[b][2]))
                          transformer = molgrid.Transform(center, 0, True)
                          transformer.forward(batch[b],batch[b])
                          gmaker.forward(center,batch[b].coord_sets[0],input_tensor[b])
                        optimizerAdam.zero_grad()

                        output = model(input_tensor[:,:24])
                        loss = criterion(output,labels)
                        predicted = torch.argmax(output,dim=1)
                        accuracy = labels.eq(predicted).sum().float() / batch_size

                        loss.backward()
                        optimizerAdam.step()

                        train_loss += loss.item()
                        train_accuracy += accuracy.item()

                        print('Epoch: {} Iteration: {} Loss: {:.4f} Accuracy: {:.2f}%'.format(
                          epoch + 1, i + 1, loss.item(), 100. * accuracy.item()))

                losses.append(train_loss / num_iterations)
                train_accuracy /= num_iterations
                accuracies.append(train_accuracy)
                print('Epoch: {} Loss: {:.4f} Accuracy: {:.2f}%'.format(epoch + 1, train_loss / num_iterations, 100. * train_accuracy / num_iterations))

                if train_accuracy > best_train_acc_max:
                    best_train_acc_max = train_accuracy
                    best_train_model_filename = f'best_{args.model}_model_{current_date}_accuracy_train_{train_accuracy:.5f}.pth'
                    full_path_to_save_train = os.path.join(bestmodel_folder_path, best_train_model_filename)
                    torch.save(model, full_path_to_save_train)
                    print('Training Accuracy increased ({:.6f} --> {:.6f}). Saving model ...'.format(
                        best_train_acc_max,
                        train_accuracy))

            elif mode == 'validation':
                model.eval()
                validation_loss = 0
                validation_accuracy = 0
                with torch.no_grad():
                  model.eval()
                  for i in range(validation_num_iterations):
                    batch = e_validate.next_batch(batch_size)
                    batch.extract_labels(float_labels)
                    centers = float_labels[:,1:]
                    labels = float_labels[:,0].long().to('cuda')
                    for b in range(batch_size):
                      center = molgrid.float3(float(centers[b][0]),float(centers[b][1]),float(centers[b][2]))
                      gmaker.forward(center,batch[b].coord_sets[0],input_tensor[b])
                    output = model(input_tensor[:,:24])
                    loss = criterion(output,labels)
                    predicted = torch.argmax(output,dim=1)
                    accuracy = labels.eq(predicted).sum().float() / batch_size

                    validation_loss += loss.item()
                    validation_accuracy += accuracy.item()
                    print('Epoch: {} Iteration: {} Loss: {:.4f} Accuracy: {:.2f}%'.format(
                      epoch + 1, i + 1, loss.item(), 100. * accuracy.item()))

                validation_losses.append(validation_loss / validation_num_iterations)
                validation_accuracy /= validation_num_iterations  # Calculate accuracy
                validation_accuracies.append(validation_accuracy)
                print('Epoch: {} Loss: {:.4f} Accuracy: {:.2f}%'.format(
                    epoch + 1, validation_loss / validation_num_iterations, validation_accuracy))

                if validation_accuracy > validation_acc_max:
                    best_validation_model_filename = f'best_{args.model}_model_{current_date}_accuracy_validation_{validation_accuracy:.5f}.pth'
                    full_path_to_save_validation = os.path.join(bestmodel_folder_path, best_validation_model_filename)
                    torch.save(model, full_path_to_save_validation)
                    print('Validation Accuracy increased ({:.6f} --> {:.6f}). Saving model ...'.format(
                        validation_acc_max,
                        validation_accuracy))
                    validation_acc_max = validation_accuracy
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f'Early stopping: No improvement in validation accuracy for {patience} epochs.')
                        break  # Stop training
    # Delete models
    del model
    del model_seresnet
    del model_cbam
    del model_cnn
    del model_resnet18
  

    # Delete optimizer and other significant tensors or variables if they exist
    del optimizerAdam
    del input_tensor
    del output
    del batch
    del centers
    del labels





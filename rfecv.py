import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import molgrid
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.feature_selection import RFECV
from sklearn.base import BaseEstimator, ClassifierMixin
from deeplearningmodels.cnn import CNNModel

class CNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_channels, num_classes=5, learning_rate=0.0001, weight_decay=0.0001):
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = CNNModel(num_classes=num_classes, input_channels=input_channels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
    def fit(self, X, y):
        self.model.train()
        X_tensor = torch.tensor(X).float()
        y_tensor = torch.tensor(y).long()
        
        for epoch in range(5):  # Simplified training loop for feature selection
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
        
        return self
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X).float()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.numpy()
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

def data_generator(path, molcache_path, batch_size=32, selected_features=None):
    e_provider = molgrid.ExampleProvider(stratify_min=0, stratify_max=5, stratify_step=1, shuffle=True, recmolcache=molcache_path, stratify_receptor=False, balanced=False)
    e_provider.populate(path)
    gmaker = molgrid.GridMaker(binary=False)
    tensor_shape = (batch_size,) + gmaker.grid_dimensions(e_provider.num_types())
    
    while True:
        float_labels = torch.zeros((batch_size, 4), dtype=torch.float32, device='cuda')
        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
        
        for _ in range(e_provider.size() // batch_size):
            batch = e_provider.next_batch(batch_size)
            batch.extract_labels(float_labels)
            centers = float_labels[:, 1:]
            labels = float_labels[:, 0].long().to('cuda')

            for b in range(batch_size):
                center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
                transformer = molgrid.Transform(center, 0, True)
                transformer.forward(batch[b], batch[b])
                gmaker.forward(center, batch[b].coord_sets[0], input_tensor[b])

            if selected_features is not None:
                yield input_tensor[:, selected_features].cpu().numpy(), labels.cpu().numpy()
            else:
                yield input_tensor.cpu().numpy(), labels.cpu().numpy()

def model_fn(input_channels=10):  # Adjust input_channels parameter
    return CNNModel(num_classes=5, input_channels=input_channels)

def weights_init(m):
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    molgrid.set_random_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    current_date = datetime.now().strftime('%Y%m%d')
    current_directory = os.path.dirname(os.path.abspath(__file__))
    bestmodel_folder_path = os.path.join(current_directory, "bestmodels")

    train_path = os.path.join(current_directory, 'dataset', 'trainfinalv0.types')
    molcache_path = os.path.join(current_directory, 'dataset', 'pdb.molcache')
    test_path = os.path.join(current_directory, 'dataset', 'testfinalv0.types')
    validation_path = os.path.join(current_directory, 'dataset', 'validatefinalv0.types')

    # Feature selection using RFECV
    train_generator = data_generator(train_path, molcache_path, batch_size=32)
    X_train, y_train = next(train_generator)  # Get the first batch from the training set

    # Initialize the model wrapper
    model_wrapper = CNNWrapper(input_channels=X_train.shape[1])

    # Use RFECV to select the best features
    rfecv = RFECV(estimator=model_wrapper, step=1, cv=3, scoring='accuracy', n_jobs=-1)
    rfecv.fit(X_train, y_train)

    # Get the selected features
    best_features = np.where(rfecv.support_)[0]
    print("Selected feature indices:", best_features)

    # Save the selected features for later use
    np.save('selected_features.npy', best_features)

    # Load selected features
    selected_features = np.load('selected_features.npy')
    print("Selected feature indices:", selected_features)

    # Initialize the model with the number of selected features
    model = model_fn(input_channels=len(selected_features))
    model.apply(weights_init)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizerAdam = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 25
    batch_size = 64
    num_iterations = 100

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_generator = data_generator(train_path, molcache_path, batch_size=batch_size, selected_features=selected_features)

        for i in tqdm(range(num_iterations)):
            X_batch, y_batch = next(train_generator)
            inputs = torch.tensor(X_batch).to(device)  # Ensure tensors are on the same device
            labels = torch.tensor(y_batch).long().to(device)

            optimizerAdam.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizerAdam.step()

            train_loss += loss.item()
            train_accuracy += (outputs.argmax(1) == labels).sum().item() / len(labels)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/num_iterations}, Accuracy: {train_accuracy/num_iterations}')

        # Validation loop
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_generator = data_generator(validation_path, molcache_path, batch_size=batch_size, selected_features=selected_features)
        num_val_iterations = 100

        with torch.no_grad():
            for i in range(num_val_iterations):
                X_batch, y_batch = next(val_generator)
                inputs = torch.tensor(X_batch).to(device)  # Ensure tensors are on the same device
                labels = torch.tensor(y_batch).long().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_accuracy += (outputs.argmax(1) == labels).sum().item() / len(labels)

        print(f'Validation Loss: {val_loss/num_val_iterations}, Validation Accuracy: {val_accuracy/num_val_iterations}')

    # Save validation results to a text file
    validation_results = f'Selected features: {best_features}\nValidation Loss: {val_loss/num_val_iterations}\nValidation Accuracy: {val_accuracy/num_val_iterations}\n'
    with open(os.path.join(current_directory, 'validation_results.txt'), 'w') as f:
        f.write(validation_results)

    # Test loop
    test_loss = 0
    test_accuracy = 0
    test_generator = data_generator(test_path, molcache_path, batch_size=batch_size, selected_features=selected_features)
    num_test_iterations = 100  # Adjust based on actual test data size

    with torch.no_grad():
        for i in range(num_test_iterations):
            X_batch, y_batch = next(test_generator)
            inputs = torch.tensor(X_batch).to(device)  # Ensure tensors are on the same device
            labels = torch.tensor(y_batch).long().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_accuracy += (outputs.argmax(1) == labels).sum().item() / len(labels)

    print(f'Test Loss: {test_loss/num_test_iterations}, Test Accuracy: {test_accuracy/num_test_iterations}')

    # Save test results to a text file
    test_results = f'Selected features: {best_features}\nTest Loss: {test_loss/num_test_iterations}\nTest Accuracy: {test_accuracy/num_test_iterations}\n'
    with open(os.path.join(current_directory, 'test_results.txt'), 'w') as f:
        f.write(test_results)

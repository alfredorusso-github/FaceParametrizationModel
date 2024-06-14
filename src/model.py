import os
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from mhm_dataset import MHMDataset
from early_stopping import EarlyStopping
from utils.parameters_to_index import parameter_to_index
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_to_load = "test_3_nose_param"
model_to_save = "test_3_nose_param"


class MHMGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Pooling blocks
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.4)

        # Full connected blocks
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 138)  # a vector of size 138 which refer to the number of parameters.

        self.relu = nn.ReLU()

    def forward(self, x):
        # Passaggio attraverso i blocchi convoluzionali e di pooling
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))

        # Flatten e passaggio attraverso i blocchi completamente connessi
        x = x.view(-1, 128 * 32 * 32)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        # Restituisce i parametri MHM
        return torch.tanh(x)


def train_one_epoch(mhm_model, train_loader, optimizer, loss_fn, epoch):
    mhm_model.train()
    total_loss = 0

    for batch_idx, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # Forward
        outputs = mhm_model(images)

        # Compute loss
        loss = loss_fn(outputs, target)
        total_loss += loss.item()

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)

    print(f'Epoch {epoch + 1}\tTrain MSE Loss: {avg_loss:.6f}')
    return avg_loss


def validate(mhm_model, val_loader, loss_fn):
    mhm_model.eval()

    total_loss = 0

    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            outputs = mhm_model(images)

            loss = loss_fn(outputs, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)

    print(f'Validation MSE Loss: {avg_loss:.6f}')
    return avg_loss


def test(mhm_model, test_loader):
    mhm_model.eval()

    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(test_loader):
            images = images.to(device)
            target = target.to(device)

            outputs = mhm_model(images)

            for i in range(images.size(0)):
                generate_graph_by_area(outputs, target, batch_idx, test_loader.batch_size, i, test_loader)

            loss = nn.MSELoss()(outputs, target)

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

    print(f'MSE: {total_loss / total_count}')


def generate_graph_by_area(outputs, target, batch_idx, batch_length, idx, test_loader):
    params_dict = {}
    path = f'../graphs/{model_to_save}'

    for params in test_loader.dataset.parameters:
        area, name = params.split("/")
        if area not in params_dict:
            params_dict[area] = set()
        params_dict[area].add(name)

    for key in params_dict.keys():
        if not os.path.exists(os.path.join(path, key)):
            os.makedirs(os.path.join(path, key))

    print(f"Saving graphs {batch_idx * batch_length + idx}")
    for area_name, params_list in params_dict.items():
        params_idx = [parameter_to_index[area_name + '/' + name] for name in params_list]

        pred = outputs[idx][params_idx].cpu().numpy()
        real = target[idx][params_idx].cpu().numpy()
        # Crea un grafico a barre per le previsioni e i valori reali
        fig, ax = plt.subplots()
        index = np.arange(len(params_idx))
        bar_width = 0.3
        opacity = 0.8

        rects1 = plt.bar(index, pred, bar_width,
                         alpha=opacity,
                         color='b',
                         label='Prediction')

        rects2 = plt.bar(index + bar_width, real, bar_width,
                         alpha=opacity,
                         color='g',
                         label='Real')

        plt.xlabel('Parameters')
        plt.ylabel('Values')
        plt.title('Comparations between prediction and real values')
        plt.xticks(index + bar_width, params_list, fontsize='small', rotation=90)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists(f'../graphs/{model_to_save}'):
            os.makedirs(f'../graphs/{model_to_save}')

        # print(f'Saving graph {area_name} number {batch_idx * batch_length + idx}')
        plt.savefig(f'../graphs/{model_to_save}/{area_name}/graph_element_{area_name}_{batch_idx * batch_length + idx}')
        plt.close(fig)


def generate_graph(outputs, target, batch_idx, batch_length, idx, test_loader):

    selected_parameters = [name.split('/')[1] for name in test_loader.dataset.parameters]
    selected_parameters_indexes = [parameter_to_index[name] for name in test_loader.dataset.parameters]

    pred = outputs[idx][selected_parameters_indexes].cpu().numpy()
    real = target[idx][selected_parameters_indexes].cpu().numpy()

    # Crea un grafico a barre per le previsioni e i valori reali
    fig, ax = plt.subplots()
    index = np.arange(len(selected_parameters))
    bar_width = 0.3
    opacity = 0.8

    rects1 = plt.bar(index, pred, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Prediction')

    rects2 = plt.bar(index + bar_width, real, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Real')

    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Comparations between prediction and real values')
    plt.xticks(index + bar_width, selected_parameters, fontsize='small', rotation=90)
    plt.legend()

    plt.tight_layout()

    if not os.path.exists(f'../graphs/{model_to_save}'):
        os.makedirs(f'../graphs/{model_to_save}')

    print(f'Saving graph number {batch_idx * batch_length + idx}')
    plt.savefig(f'../graphs/{model_to_save}/graph_element_{batch_idx * batch_length + idx}')
    plt.close(fig)


def generate_loss_graph(train_errors, val_errors):
    path = f'../graphs/{model_to_save}'
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_errors, label='Train')
    plt.plot(val_errors, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(os.path.join(path, "loss_graph"))


def save_model(mhm_model, optimizer):
    torch.save(
        {
            'model_state_dict': mhm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        f'../models/{model_to_save}.pth')


if __name__ == '__main__':
    start = time.time()

    model = MHMGenerator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    if os.path.isfile(f'../models/{model_to_load}.pth'):
        checkpoint = torch.load(f'../models/{model_to_load}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = MHMDataset(root_dir='/home/alfredo/Documenti/dataset/', type='train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    validation_dataset = MHMDataset(root_dir='/home/alfredo/Documenti/dataset/', type='validation', transform=transform)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=128, shuffle=True)

    test_dataset = MHMDataset(root_dir='/home/alfredo/Documenti/dataset/', type='test', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    loss_fn = nn.MSELoss()

    train_errors = []
    val_errors = []

    early_stopping = EarlyStopping(patience=10, delta=0.0001)
    try:
        for epoch in range(5000):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch)
            train_errors.append(train_loss)

            val_loss = validate(model, validation_loader, loss_fn)
            val_errors.append(val_loss)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    except KeyboardInterrupt:
        print("Train interrupted")

    generate_loss_graph(train_errors, val_errors)

    test(model, test_loader)
    save_model(model, optimizer)

    print(f"time: {time.time() - start}")

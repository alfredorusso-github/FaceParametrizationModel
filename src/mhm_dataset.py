from torch.utils.data import Dataset
from PIL import Image
from utils.parameters_to_index import parameter_to_index
import torch
import os


class MHMDataset(Dataset):
    def __init__(self, root_dir, type, transform=None):
        self.transform = transform
        self.samples = []
        self.parameters = []

        self.__find_files(root_dir, type)

    # noinspection PyTypeChecker
    def __find_files(self, path, type):

        if os.path.exists(os.path.join(path, type)):
            if type == 'test':
                self.__save_parameters(path)

            for element in os.listdir(os.path.join(path, type)):
                if element.endswith('.mhm'):
                    self.samples.append((os.path.join(path, type, element[:-4] + '.png'), os.path.join(path, type, element)))
        else:
            for folder in os.listdir(path):
                self.__find_files(os.path.join(path, folder), type)

    def __save_parameters(self, path):
        file = open(os.path.join(path, 'info.csv'))
        self.parameters += [parameter[:-1] for parameter in file.readlines()[1:]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mhm_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        mhm = self.load_mhm(mhm_path)

        if self.transform:
            image = self.transform(image)

        return image, mhm

    @staticmethod
    def load_mhm(mhm_path):

        # Inizializza un vettore di zeri della lunghezza corretta
        parameters = torch.zeros(len(parameter_to_index))

        file_name = ""
        with open(mhm_path, 'r') as file:
            for line in file:
                # Controlla se la linea contiene un parametro
                if line.startswith('modifier'):
                    # Estrai il nome e il valore del parametro
                    _, name, value = line.split()
                    # Aggiungi il valore al vettore dei parametri
                    # (solo se il parametro esiste nel dizionario)
                    if name in parameter_to_index:
                        parameters[parameter_to_index[name]] = float(value)

        return parameters

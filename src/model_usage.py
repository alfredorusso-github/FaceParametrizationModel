import os
import re

import torch
from PIL import Image
from torchvision import transforms
from model import MHMGenerator
from mhm_dataset import MHMDataset
from utils.parameters_to_index import parameter_to_index

STANDARD_PARAMETERS = """version v1.2.1
        camera 0.0 0.0 0.0 0.0 0.0 1.225
        modifier macrodetails/Gender 1.0
        modifier macrodetails/Caucasian 1.000000
        modifier macrodetails/African 0.000000
        modifier macrodetails/Asian 0.000000
        modifier macrodetails/Age 0.26
        modifier macrodetails-universal/Weight 0.8
        modifier macrodetails-universal/Muscle 0.500000
        modifier macrodetails-height/Height 0.500000
        modifier macrodetails-proportions/BodyProportions 0.500000"""

END = """eyes HighPolyEyes 2c12f43b-1303-432c-b7ce-d78346baf2e6
clothesHideFaces True
skinMaterial skins/default.mhmat
material HighPolyEyes 2c12f43b-1303-432c-b7ce-d78346baf2e6 eyes/materials/brown.mhmat
subdivide False"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image_and_parameters(image_path, file_path):
    image = Image.open(image_path).convert('RGB')
    file = MHMDataset.load_mhm(file_path)

    return image, file


def write_predicted_parameters_on_file(parameters, path, name):
    parameters_map = list(zip(parameter_to_index.keys(), parameters))

    file_name = f"predicted_{name[:-4]}.mhm"
    file = open(os.path.join(path, file_name), "x")

    pattern = r"\n\s+"
    str_result = f"name {file_name[:-4]}\n"
    str_result += re.sub(pattern, "\n", STANDARD_PARAMETERS) + "\n"

    for param_name, param_value in parameters_map:
        str_result += f"modifier {param_name} {param_value:.6}\n"

    file.write(str_result)
    file.close()


def get_parameters_from_mhm_file(file_path):
    file = open(file_path, 'r')

    parameters = []

    for line in file.readlines():
        if line.startswith("modifier"):
            line_split = line.split(" ")
            parameters.append(line_split[1])

    print(parameters)
    return parameters


def remove_useless_parameters(file_path):
    f = open(file_path, 'r')
    file_params = f.readlines()

    lines = []
    for line in file_params:
        if line.startswith("modifier") and 'e' not in line.split(' ')[2]:
            lines.append(line)
        if line.startswith("name") or line.startswith("camera"):
            lines.append(line)

    f.close()

    f = open(file_path, "w")
    for line in lines:
        f.write(line)

    pattern = r"\n\s+"
    string_to_write = re.sub(pattern, "\n", END)
    f.write(string_to_write)

    f.close()


def process_images(path, mhm_model, transform):
    mhm_model.eval()

    for file in os.listdir(path):
        if file.endswith("png"):
            img = Image.open(os.path.join(path, file)).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0).to(device)

            predictions = mhm_model(img)
            with torch.no_grad():
                predictions_cpu = predictions[0].cpu().numpy()

            write_predicted_parameters_on_file(predictions_cpu, path, file)

            name = f"predicted_{file[:-4]}.mhm"
            remove_useless_parameters(os.path.join(path, name))


if __name__ == '__main__':
    mhm_model = MHMGenerator().to(device)
    checkpoint = torch.load('../models/model_random_25_75.pth')
    mhm_model.load_state_dict(checkpoint['model_state_dict'])
    # summary(mhm_model, input_size=(3, 128, 128))

    images_path = '/home/alfredo/Scaricati/Re R R Appuntamento di oggi/test_model/'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    process_images(images_path, mhm_model, transform)

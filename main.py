import pickle
import os
import open_clip
import copy
import random
import ast
import torch
import json
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from decouple import config
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from model import evaluate_model, train_model_camelyon, train_model_cifar, train_model_cifar_LoRa, train_model_cifar_LoRa_BNN, evaluate_model_freeze
from utils import generate_results, Paths, generate_and_save_plot, bar_plot_diff, block_diff, generate_particles
from preprocessor import load_data_camelyon, load_data_cifar
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from bayes_wrap import BayesWrap


"""
# LoRa
from opendelta import Visualization
from opendelta import LowRankAdapterModel, AdapterModel
from opendelta import LoraModel # use lora as an example, others are same
"""

random.seed(229)

print("The opt value is: ", config('opt'))
''' -----------------------   Set path ------------------------------'''
paths = Paths(config)
paths.create_path()


''' -----------------------   loading CLIP ViT ------------------------------'''
device = f"cuda:{config('device')}" if torch.cuda.is_available() else "cpu"

# mdl, preprocess = clip.load('ViT-B/32', device)
# this uses the pretrained on Laion 2B
mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


download_path = os.path.expanduser("/media/rokny/DATA1/Afshar/data")
if config('dataset_name').upper() == "CAMELYON":

    dataset = get_dataset(dataset="camelyon17", download=True, root_dir=download_path)
    train_data = dataset.get_subset(
        "train",
        transform=preprocess
    )

    val_data = dataset.get_subset(
        "val",
        transform=preprocess
    )

    test_data = dataset.get_subset(
        "test",
        transform=preprocess
    )
    print('camelyon loaded')
    train_loader, validation_loader, test_loader = load_data_camelyon(preprocess, train_data, val_data, test_data, device)


elif config('dataset_name').upper() == "CIFAR10":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("/media/rokny/DATA1/Afshar/Data/" + "cifar-10-batches-py")
    train = CIFAR10(root, download=True, train=True)
    test = CIFAR10(root, download=True, train=False, transform=preprocess)
    print(f'len test: {len(test)}')

    print('cifar loaded')
    train_loader, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

elif config('dataset_name').upper() == "CIFAR100":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("/media/rokny/DATA1/Afshar/Data/" + "cifar-100-batches-py")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)
    print(f'len test: {len(test)}')

    print('cifar 100 loaded')
    train_loader, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)



''' -----------------------   Training the models   ----------------------- '''

print('--------------------------Traning has been started-------------------- ')
for i,noise_std in enumerate(ast.literal_eval(config('noise_std_list'))):

    print(f"training model {i} with noise {noise_std} has been started")

    device = f"cuda:{config('device')}" if torch.cuda.is_available() else "cpu"
    mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


    if config('dataset_name').upper() == "CAMELYON":
        loss_values, loss_values_val, all_scores, all_labels = train_model_camelyon(mdl, train_loader, validation_loader, test_loader, noise_std, i, config)
    elif  config('dataset_name').upper() in ["CIFAR10", "CIFAR100"]:
        print("Config:", config('dataset_name'), config('run_name'), config("batch_size"), config("num_epochs"), config("opt"), config("decay_rate"))
        print(f"Training LoRa with dataset:{str(config('dataset_name'))}")
        loss_values, loss_values_val, all_scores, all_labels = train_model_cifar(mdl, train_loader, validation_loader, test_loader, noise_std, i, config)
        # loss_values, loss_values_val, all_scores, all_labels = train_model_cifar_LoRa(mdl, train_loader, validation_loader, test_loader, noise_std, i, config)
        # loss_values, loss_values_val, all_scores, all_labels = train_model_cifar_LoRa_BNN(mdl, train_loader, validation_loader, test_loader, noise_std, i, config)
    print(f'training model_{i}_noise_{noise_std} is completed')
    
    run_name = config('run_name')
    save_path = "Model/" + f"Loss_model_{i}_noise_std_{noise_std}_{run_name}.png"
    generate_and_save_plot(loss_values, loss_values_val, save_path) 
    generate_results(all_scores, all_labels, noise_std, i, paths = paths.path_results)
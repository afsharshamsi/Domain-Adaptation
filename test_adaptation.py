
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
import deeplake
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from decouple import config
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets import CIFAR10
from model import evaluate_model, train_model_camelyon, train_model_cifar, evaluate_model_freeze, evaluate_model_cam_ensemble_freeze, averaging_model, tent_cifar
from utils import generate_results, Paths, generate_and_save_plot, bar_plot_diff, block_diff, generate_particles
from preprocessor import load_data_camelyon, load_data_cifar, load_data_places
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from bayes_wrap import BayesWrap, generate_freezed_particles, train_model_wrap_cifar, generate_lora_particles



random.seed(2295)

''' -----------------------   Set path ------------------------------'''
paths = Paths(config)
paths.create_path()


''' -----------------------   loading CLIP ViT ------------------------------'''
device = "cuda" if torch.cuda.is_available() else "cpu"

# mdl, preprocess = clip.load('ViT-B/32', device)
mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

download_path = os.path.expanduser("/media/rokny/DATA1/Afshar/data")
if config('dataset_name').upper() == "CAMELYON":

    dataset = get_dataset(dataset="camelyon17", download=True,  root_dir=download_path)
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
    trainloaders = [torch.utils.data.DataLoader(train_data, batch_size=int(config('batch_size')), shuffle=True) for i in range(int(config('opt')))]
    valloader = torch.utils.data.DataLoader(val_data, batch_size=int(config('batch_size')), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=int(config('batch_size')), shuffle=False) 
elif config('dataset_name').upper() == "CIFAR10":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("/media/rokny/DATA1/Afshar/Data/" + "cifar-10-batches-py")
    train = CIFAR10(root, download=True, train=True)
    test = CIFAR10(root, download=True, train=False, transform=preprocess)
  
    # corrupted_testset = np.load("Data/shot_noise.npy")
    # lbls = np.load("Data/labels.npy")
    # test.data = corrupted_testset
    # test.targets = lbls
    # test.transform = preprocess

    # print(f'len test zahra: {len(test)}')


    print('cifar10 loaded')
    # trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

elif config('dataset_name').upper() == "CIFAR100":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("/media/rokny/DATA1/Afshar/Data/" + "cifar-100-batches-py")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)

    print('cifar100 loaded')
    trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)


elif config('dataset_name').upper() == "DOMAINNET":

    ''' -----------------------   Loading the Data   ----------------------- '''
    train_data = deeplake.load("hub://activeloop/domainnet-real-train")
    test_data = deeplake.load("hub://activeloop/domainnet-real-test")

    print('Domainnet has been loaded')
    print(f'len train is {len(train_data)}')
    print(f'len test is {len(test_data)}')

    trainloaders, validation_loader, test_loader = load_data_places(preprocess, train_data, test_data, test_data, device)


#####################################################################################################
###############################  Domain Adaptation CIFAR-10-C  ############################################
if config('dataset_name').upper() == "CIFAR10":

    mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model_address = [f for f in os.listdir('./nmdl/') if f[-3:]=='.pt']
    print(f'number of checkpoints is {len(model_address)}')

    average_model, ensemble = averaging_model(model_address)

    particles=[]
    particles.append(average_model.to(device))

    delta_models = generate_lora_particles(particles)
    noise_std = [0]
    # i = 0

    corrupted_address = [f for f in os.listdir("./Data/") if f[-4:]=='.npy']

    # print(f'corrupted is {corrupted_address}')
    performance=[]
    for i, corr in enumerate(corrupted_address):
        if corr != "labels.npy":
            print(f'noise is {corr.split(".")[0]}')
            perf=[]
            corrupted_testset = np.load("Data/" + corr)
            lbls = np.load("Data/labels.npy")
            test.data = corrupted_testset
            test.targets = lbls
            test.transform = preprocess
            print(f'len {corr} is {len(test)}')
            trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)
            perf.append(corr.split(".")[0])

            all_scores, all_labels = evaluate_model_freeze(average_model, test_loader, device)
            # all_scores, all_labels= evaluate_model_cam_ensemble_freeze(delta_models, test_loader, device)
            accuracy = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
            perf.append(accuracy)

            test_load = []
            test_load.append(test_loader)
            ten_acc = tent_cifar(delta_models, test_load, test_loader, noise_std, config)
            perf.append(ten_acc)

            performance.append(perf)
        
        
        
    performance_path = f"Model/tent_cifar10.json"
    with open(performance_path, 'w') as fp:
        json.dump(performance, fp, indent=2)
    print(performance)

#####################################################################################################
###############################  Domain Adaptation Domainnet ############################################

if config('dataset_name').upper() == "DOMAINNET":

    mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model_address = [f for f in os.listdir('./nmdl/') if f[-3:]=='.pt']
    print(f'number of checkpoints is {len(model_address)}')

    average_model, ensemble = averaging_model(model_address)

    particles=[]
    particles.append(average_model.to(device))

    delta_models = generate_lora_particles(particles)

    noise_std = [0]
    i=0


    ## print(f'corrupted is {corrupted_address}')
    domainnet=['clip', 'paint', 'sketch', 'info', 'quick']

    performance=[]
    for i, corr in enumerate(domainnet):
            print(f'data is {corr}')
            perf=[]

            test = deeplake.load(f"hub://activeloop/domainnet-{corr}-test")
            print(f'len {corr} is {len(test)}')
            trainloaders, validation_loader, test_loader = load_data_places(preprocess, train_data, test, test, device)
            perf.append(corr)

            all_scores, all_labels = evaluate_model_freeze(average_model, test_loader, device)
            accuracy = generate_results(all_scores, all_labels, noise_std[0], i, paths = paths.path_results)
            perf.append(accuracy)

            test_load = []
            test_load.append(test_loader)
            ten_acc = tent_cifar(delta_models, test_load, test_loader, noise_std, config)
            perf.append(ten_acc)

            performance.append(perf)
            
        
        
    performance_path = f"Model/tent_domainnet.json"
    with open(performance_path, 'w') as fp:
        json.dump(performance, fp, indent=2)
    print(performance)

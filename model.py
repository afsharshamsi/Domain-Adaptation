import torch
import math
import copy
import open_clip
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from decouple import config
from utils import add_noise_to_parameters
import numpy as np
from sklearn.metrics import accuracy_score
import os
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
from bayes_wrap import BayesWrap, LP1WithLora, LP1WithLora_SVGD, SVGD
import torch.optim.lr_scheduler as lr_scheduler
from utils import cosine_lr

from opendelta.utils.inspect import inspect_module_statistics
from bigmodelvis import Visualization

"""
from opendelta import Visualization
from opendelta import LowRankAdapterModel, AdapterModel
from opendelta import LoraModel # use lora as an example, others are same
"""

device = f"cuda:{config('device')}" if torch.cuda.is_available() else "cpu"

def train_model_camelyon(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):
    """this function trains the model and returns the losses and the entropies of the test set"""

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()


    opt = config('opt')
    model = BayesWrap(NET, opt)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=float(config('learning_rate')))


    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, step = 0., 0.
        for i, (img, labels, metadata) in enumerate(train_loader):
            
            img, labels = img.cuda(), labels.cuda()

            optimizer.zero_grad()

            kwargs = {"return_entropy": False}
            logits, soft_out = model(img, **kwargs)

            loss = criterion(logits, labels)
            loss.backward()
            model.update_grads()

            losses += loss.item()
            step += 1
            
            optimizer.step()     
            optimizer.zero_grad()
            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        

        loss_epoch = losses / step
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, step2 = 0., 0.
        for img, text, metadata in validation_loader:
            img, text = img.cuda(), text.cuda()
            logits, soft_out = model(img, **kwargs)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(soft_out, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            step2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / step2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        # Save checkpoint if the current validation loss is the best
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"


    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs=[], []
    i = 0
    with torch.no_grad():

        for images, labels, metadata in test_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out = model(img, **kwargs)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy()) 
            all_softs.extend(soft_out.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            all_entropies.extend(entropies.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1

    all_scores_train = []
    all_labels_train = []
    all_entropies_train=[]

    i = 0
    with torch.no_grad():

        for images, labels, metadata in train_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out = model(img, **kwargs)

            predicted = torch.argmax(logits, dim=1)
            all_scores_train.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels_train.extend(labels.numpy())  # Extend the list with true labels
            all_entropies_train.extend(entropies.cpu().numpy())
            print(f'\r calculating entropies for train {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_scores_train = np.array(all_scores_train).tolist()
        all_labels_train = np.array(all_labels_train).tolist()
        all_entropies_train = np.array(all_entropies_train).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test": all_softs,
                     "all_labels_train": all_labels_train,
                     "all_scores_train": all_scores_train,
                     "all_entropies_train": all_entropies_train}

    run_name = config('run_name')
    labels_info_path = f"Results/{run_name}/entropies.json"
    if not os.path.exists(f"Results/{run_name}"):
        os.makedirs(f"Results/{run_name}")
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    # Save best model
    # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"
    # torch.save(best_model, best_model_path)

    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels




def train_model_cifar(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()

    opt = config('opt')
    model = BayesWrap(NET, opt)

    trainable = count_trainable_parameters(model)
    print("trainable parameters of normal model: ", trainable)
    model = model.cuda()

    # temperature_factor = float(config('temperature_factor'))
    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=float(config('learning_rate')))

    

    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, step = 0., 0.
        for i, (img, labels) in enumerate(train_loader):
            
            img, labels = img.cuda(), labels.cuda()

            optimizer.zero_grad()

            kwargs = {"return_entropy": False}
            logits, soft_out = model(img, **kwargs)

            loss = criterion(logits, labels)
            loss.backward()
            model.update_grads()

            losses += loss.item()
            step += 1
            
            optimizer.step()     
            # optimizer.zero_grad()
            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        
        

        loss_epoch = losses / step
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, step2 = 0., 0.
        for img, text in validation_loader:
            img, text = img.cuda(), text.cuda()
            logits, soft_out = model(img, **kwargs)
        
            loss_val = criterion(soft_out, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(logits, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            step2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / step2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"


    model.eval()  

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out, stds = model(img, **kwargs)

            predicted = torch.argmax(soft_out, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(soft_out.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1

    all_scores_train = []
    all_labels_train = []
    all_entropies_train=[]

    i = 0
    with torch.no_grad():

        for images, labels in train_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out, _ = model(img, **kwargs)

            predicted = torch.argmax(soft_out, dim=1)
            all_scores_train.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels_train.extend(labels.numpy())  # Extend the list with true labels
            all_entropies_train.extend(entropies.cpu().numpy())
            print(f'\r calculating entropies for train {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_stds = np.array(all_stds).tolist()
        all_scores_train = np.array(all_scores_train).tolist()
        all_labels_train = np.array(all_labels_train).tolist()
        all_entropies_train = np.array(all_entropies_train).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds,
                     "all_labels_train": all_labels_train,
                     "all_scores_train": all_scores_train,
                     "all_entropies_train": all_entropies_train}

    run_name = config('run_name')
    labels_info_path = f"Results/{run_name}/entropies.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)



    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels


def count_trainable_parameters(model, modified_modules=None): 
    """
    Counts the no of delta parameters, trainable parameters and total parameters in a LoRa model.

    Args:
        model: A LoRa model.
        modified_modules: A list of modified modules in the model.
    
    Returns:
        delta_parameters: The number of delta parameters in the model.
        trainable_parameters: The number of trainable parameters in the model.
        total_parameters: The total number of parameters in the model.
    """
    stat = inspect_module_statistics(model, modified_modules)
    delta_parameters = stat["delta_parameters"]
    trainable_parameters = stat["trainable_parameters"]
    total_params = stat["total_parameters"]
    return trainable_parameters, delta_parameters, total_params


"""
from torchsummary import summary
"""

def train_model_cifar_LoRa(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):
    """train a model using LoRa"""

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    net = ImageClassifier(image_encoder, classification_head)
    net.freeze_head()

    opt = config('opt')
    # model = BayesWrap(NET, opt)
    model = LP1WithLora(net)

    # print how much trainable paramters of LoRa model
    # summary(model, input_size=(3, 32, 32))
    trainable = count_trainable_parameters(model)
    print("trainable parameters of LoRa model: ", trainable)

    model = model.to(device)

    # temperature_factor = float(config('temperature_factor'))
    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params, lr=float(config('learning_rate')))
    # change to Adam optimizer
    # optimizer = optim.Adam(params, lr=float(config('learning_rate')))
    optimizer = torch.optim.AdamW(params, lr=float(config('learning_rate')), weight_decay=float(config('Weight_decay')))
    
    scheduler = cosine_lr(
        optimizer,
        float(config('learning_rate')),
        int(config("warmup_length")),
        int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
    )
    

    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, tep = 0., 0.
        for i, (img, labels) in enumerate(train_loader):

            step = (
                i // int(config('num_grad_accumulation'))
                + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
            )
            
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()

            # kwargs = {"return_entropy": False}
            logits = model(img)

            loss = criterion(logits, labels)
            loss.backward()
            # model.update_grads()

            losses += loss.item()
            tep += 1
            
            if (i + 1) % int(config('num_grad_accumulation')) == 0:
                scheduler(step)

                optimizer.step()     
            # optimizer.zero_grad()
            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        
        

        loss_epoch = losses / tep
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, tep2 = 0., 0.
        for img, text in validation_loader:
            img, text = img.to(device), text.to(device)
            logits = model(img)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(logits, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            tep2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / tep2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"
        
        # save the best model
        if epoch == best_epoch:
            print("saving the best model")
            if not os.path.exists(f"Model/{config('run_name')}"):
                os.makedirs(f"Model/{config('run_name')}")
            torch.save(model.state_dict(), f"Model/{config('run_name')}/best_model_noise_std_{noise_std}.pt")
        


    model.eval()  

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            kwargs = {"return_entropy": True}
            # logits, entropies, soft_out, stds = model(img, **kwargs)
            logits = model(img)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            # all_entropies.extend(entropies.cpu().numpy())
            # all_softs.extend(soft_out.cpu().numpy())
            # all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1

    all_scores = np.array(all_scores).tolist()
    all_labels = np.array(all_labels).tolist()


    # all_scores_train = []
    # all_labels_train = []
    # all_entropies_train=[]



    #     # Convert the lists of scores and labels to NumPy arrays
    #     all_scores = np.array(all_scores).tolist()
    #     all_labels = np.array(all_labels).tolist()
    #     # all_entropies = np.array(all_entropies).tolist()
    #     # all_softs = np.array(all_softs).tolist()
    #     # all_stds = np.array(all_stds).tolist()


    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores 
                    #  "all_entropies_test":  all_entropies,
                    #  "all_softs_test":  all_softs,
                    #  "all_std_test": all_stds,

                     }
    run_name = config('run_name')
    labels_info_path = f"Results/{run_name}/entropies.json"
    if not os.path.exists(f"Results/{run_name}"):
        os.makedirs(f"Results/{run_name}")
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)



    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels



def train_model_cifar_LoRa_BNN(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):
    """train a model using LoRa applying SVGD for BNN approximation"""

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    net = ImageClassifier(image_encoder, classification_head)
    net.freeze_head()

    opt = config('opt')
    
    model = LP1WithLora_SVGD(net, opt)
    print("\nArchitecture recieved from LP1WithLoRa_SVGD:")
    Visualization(model).structure_graph()
    trainable, delta, total = count_trainable_parameters(model, None)
    print(f"[LORA_SVGD FINAL] Trainable parameters: {trainable}, delta parameters: {delta}, total parameters: {total}")

    model = model.to(device)

    ## freeze all parameters except deltas
    # for name, param in model.named_parameters():
    #     if 'lora' not in name:
    #         param.requires_grad = False

    # Visualize the model
    print("\nFinal model architecture, FROZEN:")
    Visualization(model).structure_graph()
    trainable, delta, total = count_trainable_parameters(model, None)
    print(F"[LORA_SVGD FROZEN FINAL] Trainable parameters: {trainable}, delta parameters: {delta}, total parameters: {total}")

    # temperature_factor = float(config('temperature_factor'))
    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=float(config('learning_rate')))
    optimizer = torch.optim.AdamW(params, lr=float(config('learning_rate')), weight_decay=float(config('decay_rate')))

    scheduler = cosine_lr(
        optimizer,
        float(config('learning_rate')),
        int(config("warmup_length")),
        int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
    )    

    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, tep = 0., 0.
        for i, (img, labels) in enumerate(train_loader):
            step = (
                i // int(config('num_grad_accumulation'))
                + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
            )            
            
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()

            # kwargs = {"return_entropy": True}
            # logits, soft_out = model(img)
            loss = model.get_losses(img, labels, criterion)
            # print("Logits", len(logits), logits[0].shape, logits[1].shape)
            
            loss.backward()
            model.update_grads()

            losses += loss.item()
            tep += 1

            if (i + 1) % int(config('num_grad_accumulation')) == 0:
                scheduler(step)
                optimizer.step()  

            # optimizer.step()            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        
        

        loss_epoch = losses / tep
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, tep2 = 0., 0.
        for img, text in validation_loader:
            img, text = img.to(device), text.to(device)
            logits,_  = model(img)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(logits, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            tep2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / tep2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"
            print("Saving the model...")
            for i, particle in enumerate(model.particles):
                torch.save(particle.state_dict(), f'Model/best_model_{i}_noise_std_{noise_std}.pt')

    model.eval()  

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            kwargs = {"return_entropy": True}
            # logits, entropies, soft_out, stds = model(img, **kwargs)
            logits, entropies = model(img)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            all_entropies.extend(entropies.cpu().numpy())
            # all_softs.extend(soft_out.cpu().numpy())
            # all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1


        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies
                    #  "all_softs_test":  all_softs,
                    #  "all_std_test": all_stds,

                     }
    run_name = config('run_name')
    labels_info_path = "Results/{run_name}"
    if not os.path.exists(labels_info_path):
        os.makedirs(labels_info_path)
    labels_info_path = os.path.join(labels_info_path, 'entropies.json')
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)



    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels



def evaluate_model(model, test_loader, text_inputs, device):

    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            # img, text = img.cuda(), text.cuda()
            img_feats = model.encode_image(img)
            text_feats = model.encode_text(text_inputs)

            # img_feats /= img_feats.norm(dim=-1, keepdim=True)
            # text_feats /= text_feats.norm(dim=-1, keepdim=True)
            logits = torch.matmul(img_feats, text_feats.T)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
    return  all_scores, all_labels    




def evaluate_model_freeze(model, test_loader, device):
    
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    i = 0
    with torch.no_grad():
        if config('dataset_name').upper() == "CAMELYON":
            for images, labels,_ in test_loader:
            
                img , text = images.to(device), labels.to(device)
                logits = model(img)

                predicted = torch.argmax(logits, dim=1)
                all_scores.extend(predicted.cpu().numpy())  
                all_labels.extend(labels.numpy())  
                print(f'\r {i}', end='')
                i +=1
        else:
            for images, labels in test_loader:
            
                img , text = images.to(device), labels.to(device)
                logits = model(img)

                predicted = torch.argmax(logits, dim=1)
                all_scores.extend(predicted.cpu().numpy())  
                all_labels.extend(labels.numpy()) 
                print(f'\r {i}', end='')
                i +=1
      
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
    return  all_scores, all_labels    


def evaluate_model_cam_ensemble_freeze(ensemble, test_loader, device):

    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_entropies, all_softs, all_stds = [], [], [], []
    i = 0
    with torch.no_grad():
      
        for images, labels,_ in test_loader:
            
            img , text = images.to(device), labels.to(device)
 
            logits = [] 
            softs, entropies = [],[]
            for model in ensemble:
 
                model = model.cuda()
                l = model.backbone_model(img)
                sft = torch.softmax(l, 1)

                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)
                
            logits = torch.stack(logits).mean(0)
            stds = torch.stack(softs).std(0)
            softs = torch.stack(softs).mean(0)
            entropies = torch.stack(entropies).mean(0)

            predicted = torch.argmax(softs, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(softs.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())            

            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        # all_scores = np.array(all_scores)
        # all_labels = np.array(all_labels)

        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_stds = np.array(all_stds).tolist()


    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds
                                                }

    labels_info_path = f"Results/entropies_model_cam.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    return  all_scores, all_labels 

#-----------------------------------------------------------------------------------------------------------------
def averaging_model(model_address):

    ensemble=[]
    for i in range(len(model_address)):
        mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        # mdl_addr = f'nmdl/mdl_{i}.pt'
        classification_head = get_classification_head()
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)
        net = ImageClassifier(image_encoder, classification_head)
        net.freeze_head()

    
        model_new = copy.deepcopy(net)
        fine_tuned_weights = torch.load("./nmdl/"+ model_address[i])
        model_new.load_state_dict(fine_tuned_weights)
        ensemble.append(model_new)
        print(f'model {i} is loaded from {model_address[i]}')


    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    classification_head = get_classification_head()
    image_encoder = ImageEncoder(model)#, keep_lang=False)
    net = ImageClassifier(image_encoder, classification_head)
    net.freeze_head()


    average_model = copy.deepcopy(net)

    state_dicts = [mdel.state_dict() for mdel in ensemble]

    average_state_dict = {}
    num_models = len(ensemble)
    # print(f'number of models are {num_models}')


    # coefficients = [n, m , k]
    # print(f'len coefficient is {len(coefficients)}')



    for key in ensemble[0].state_dict():
        average_state_dict[key] =sum(state_dict[key] for state_dict in state_dicts) / num_models
        # average_state_dict[key] = sum([coeff * state_dict[key] for coeff, state_dict in zip(coefficients, state_dicts)])#/ len(coefficients)

    average_model.load_state_dict(average_state_dict)

    print('The averaged model will be used for comparison')
    print("")   

    return average_model, ensemble

#-----------------------------------------------------------------------------------------------


def tent_cifar(particles, trainloaders, valloader, noise_std, config):
    h_kernel = 0
    # criterion = nn.CrossEntropyLoss()

    best_losses = [float('inf')] * len(particles)
    best_val_accuracy = [float('inf')] * len(particles)

    learning_rates = [0.0008]#, 0.0009, 0.0012, 0.001, 0.0008]

    optimizers = [optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=float(config('Weight_decay'))) for model, lr in zip(particles, learning_rates)]


    for epoch in range(int(config('num_epochs'))):
        
        accumulated_losses = [0.0] * len(particles)
        num_batches = len(next(iter(trainloaders)))

        for j,batches in enumerate(zip(*trainloaders)):
            inputs_list = [batch[0] for batch in batches]
            targets_list = [batch[1] for batch in batches]
            for i, (model, imgs, lbls) in enumerate(zip(particles, inputs_list, targets_list)):
                
                scheduler = cosine_lr(
                                            optimizers[i],
                                            learning_rates[i],
                                            int(config("warmup_length")),
                                            int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
                                        )

                step = (
                            i // int(config('num_grad_accumulation'))
                            + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
                                                            )
                imgs, labels = imgs.cuda(), lbls.cuda()

                optimizers[i].zero_grad()

                logits = model.backbone_model(imgs)
                sft = torch.softmax(logits, 1)
                ent =  (-sft * torch.log(sft + 1e-8)).sum(1)

                loss = ent.mean(0)
                loss.backward()
                accumulated_losses[i] += loss.item()
            print(f'\rProcessing batch {j+1}/{num_batches}', end='')
       
            # h_kernel =update_gradiants(particles, h_kernel)

            for optimizer in optimizers:
                scheduler(step)
                optimizer.step()

        average_losses = [loss_sum / num_batches for loss_sum in accumulated_losses]
        print(" ")
        for i, avg_loss in enumerate(average_losses):
            print(f"Epoch {epoch}, Model {i}, Average Epoch Loss: {avg_loss}")

        all_scores, all_labels=[],[]
        with torch.no_grad():
            for i,model in enumerate(particles):

                correct = 0
                total = 0
                losses_eval, step2 = 0., 0.
                for img, lbls in valloader:
                    img, label = img.cuda(), lbls.cuda()

                    logits = model.backbone_model(img)
                    # loss_val = criterion(logits, label)
                    # losses_eval += loss_val.item()
                    _, predicted = torch.max(logits, 1)
                    all_scores.extend(predicted.cpu().numpy())  
                    all_labels.extend(lbls.numpy())                     
                    # total += label.size(0)
                    # correct += (predicted == label).sum().item()
                    step2 += 1

                # accuracy = correct / total
                accuracy = accuracy_score(all_labels, all_scores)
                print(f'[Epoch: {epoch}], val_acc_{i}: {accuracy:.4f}')
                
    print('finished')        
    return accuracy

# def tent_cifar(particles, trainloaders, valloader, noise_std, config):
#     h_kernel = 0
#     # criterion = nn.CrossEntropyLoss()

#     best_losses = [float('inf')] * len(particles)
#     best_val_accuracy = [float('inf')] * len(particles)

#     learning_rates = [float(config('learning_rate'))]#, 0.0009, 0.0012, 0.001, 0.0008]

#     optimizers = [optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=float(config('Weight_decay'))) for model, lr in zip(particles, learning_rates)]


#     for epoch in range(int(config('num_epochs'))):
        
#         accumulated_losses = [0.0] * len(particles)
#         num_batches = len(next(iter(trainloaders)))

#         for j,batches in enumerate(zip(*trainloaders)):
#             inputs_list = [batch["images"] for batch in batches]
#             targets_list = [batch["labels"] for batch in batches]
#             for i, (model, imgs, lbls) in enumerate(zip(particles, inputs_list, targets_list)):
                
#                 scheduler = cosine_lr(
#                                             optimizers[i],
#                                             learning_rates[i],
#                                             int(config("warmup_length")),
#                                             int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
#                                         )

#                 step = (
#                             i // int(config('num_grad_accumulation'))
#                             + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
#                                                             )
#                 imgs, labels = imgs.cuda(), lbls.cuda()

#                 optimizers[i].zero_grad()

#                 logits = model.backbone_model(imgs)
#                 sft = torch.softmax(logits, 1)
#                 ent =  (-sft * torch.log(sft + 1e-8)).sum(1)

#                 loss = ent.mean(0)
#                 loss.backward()
#                 accumulated_losses[i] += loss.item()
#             print(f'\rProcessing batch {j+1}/{num_batches}', end='')
       
#             # h_kernel =update_gradiants(particles, h_kernel)

#             for optimizer in optimizers:
#                 scheduler(step)
#                 optimizer.step()

#         average_losses = [loss_sum / num_batches for loss_sum in accumulated_losses]
#         print(" ")
#         for i, avg_loss in enumerate(average_losses):
#             print(f"Epoch {epoch}, Model {i}, Average Epoch Loss: {avg_loss}")

#         all_scores, all_labels= [], []
#         with torch.no_grad():
#             for i,model in enumerate(particles):

#                 correct = 0
#                 total = 0
#                 losses_eval, step2 = 0., 0.
#                 for j, (img, lbls) in enumerate(valloader):
#                     img, label = img.cuda(), lbls.cuda()

#                     logits = model.backbone_model(img)
#                     # loss_val = criterion(logits, label)
#                     # losses_eval += loss_val.item()
#                     _, predicted = torch.max(logits, 1)
#                     all_scores.extend(predicted.cpu().numpy())  
#                     all_labels.extend(lbls.numpy()) 
#                     step2 += 1
#                     print(f'\r {j}', end='')
#                 print(" ")    
#                 accuracy = accuracy_score(all_labels, all_scores)
#                 print(f'[Epoch: {epoch}], val_acc_{i}: {accuracy:.4f}')
                
#     print('finished')        
#     return accuracy







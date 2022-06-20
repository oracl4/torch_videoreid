import os
import glob
import random
import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

import wandb

from Model import Model
import test_PRID_old

# ===================================================================================================
# All parameter, change before training

# WanDB logging
wandb_flag = True

# CUDA Device
cuda_device = torch.device("cuda:1")

# Training parameters
learning_rate = 0.000002
n_epochs = 120
batch_size = 1

# Network parameters
n_features = 512    # Number of features from feature extractor
n_hidden = 512      # hidden layer num of LSTM
n_frames = 18       # Number of temporal frames
n_partitions = 7    # Number of partitions / spatial frames (4 Local Pooling + 2 Local Pooling + 1 Global Pooling)  
n_head = 8          # Number of self-attention head
n_cluster = 38      # Number of cluster (for compatibility matrix only ?)

#### Mars Dataset
dataset_name = "PRID"
n_class = 89
experiment_name = "PEAM_OldFeature"

# Global features path
globalfeat_path = '../../features/input/PRID/previous/train/train_glofeat.mat'

# New global features path
localfeat2_path = '../../features/input/PRID/previous/train/train_split_2/'
localfeat4_path = '../../features/input/PRID/previous/train/train_split_4/'

# Evaluation path
evaluation_path =  '../evaluation/PRID'

# ===================================================================================================

# WanDB logging
if wandb_flag:

    # Initialize the WanDB
    wandb.init(project="person-reid", entity="oracl4")

    # Save the parameter
    wandb.config = {
        "dataset_name": dataset_name,
        "experiment" : experiment_name,
        "learning_rate" : learning_rate,
        "epochs" : n_epochs,
        "batch_size" : batch_size,
        "n_features" : n_features,
        "n_hidden" : n_hidden,
        "n_frames" : n_frames,
        "n_partitions" : n_partitions,
        "n_head" : n_head,
        "n_cluster" : n_cluster
    }


# Work directory
work_dir = "../../work_dir"
work_dir = os.path.join(work_dir, dataset_name, experiment_name)

# Model and tensorboard dir
model_dir = os.path.join(work_dir, "model")
tensorboard_dir = os.path.join(work_dir, "tensorboard")

# Tensorboard
writer = SummaryWriter(tensorboard_dir)

# Create the directory
if not os.path.exists(work_dir):
	os.makedirs(work_dir)

if not os.path.exists(model_dir):
	os.makedirs(model_dir)

if not os.path.exists(tensorboard_dir):
	os.makedirs(tensorboard_dir)

# Load global features
grf = sio.loadmat(globalfeat_path)
grf = grf['glofeat']

# Load the local features List
trainlist2 = sorted(glob.glob(localfeat2_path+'/*.mat'))
trainlist4 = sorted(glob.glob(localfeat4_path+'/*.mat'))
train_num = len(trainlist2)

# Load the labels 0 - train_num/2 (89)
label = np.arange(0, train_num/2)
cls_list = np.repeat(label, 2)

# Encode the ground truth
class_encoding = np.eye(n_class)

# Class label
raw_labels = np.zeros((train_num, n_frames, 1))

# One hot encoding label
raw_labels_ohe = np.zeros((train_num, n_frames, n_class))

# Encode the ground truth
for i in range(0, train_num):

    label_id = int(cls_list[i])
    
    lbs = np.expand_dims(label_id, 0)
    lbs = np.tile(lbs, (n_frames, 1))
    
    cl = np.expand_dims(class_encoding[label_id, :], 0)
    cl = np.tile(cl, (n_frames,1))
    
    raw_labels[i,:,:] = lbs
    raw_labels_ohe[i,:,:] = cl

raw_labels = raw_labels.astype('float')
raw_labels_ohe = raw_labels_ohe.astype('float')

# ===================================================================================================
# Model creation

TrainingModel = Model(batch_size, n_features, n_hidden, n_class, n_frames, n_partitions, n_head, n_cluster, cuda_device)

# Optimizer with L2 regularization
optimizer = torch.optim.Adam(params=TrainingModel.parameters(), lr=learning_rate, weight_decay=1e-5)

# Learning rate scheduler
LR_decayRate = 0.5
LR_Scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=LR_decayRate)

print(TrainingModel)

# ===================================================================================================
# Training Loop

# Best test mAP
best_eval = 0.0

print("##########   Start Training")

for epoch in range(0, n_epochs+1):

    # Calculate the number of iterations based on the dataset size and batch size
    n_iter = np.arange(0, int(train_num/batch_size))

    # Arrange the batch size
    n_batchs = np.arange(0, train_num)
    np.random.shuffle(n_batchs)

    # Epoch loss placeholder
    epoch_loss = []
    epoch_crf_loss =[]

    # Dataset batch loop
    for batch in tqdm(n_iter):

        # Get the idx based on the batch size
        ix = n_batchs[batch:(batch+batch_size)]

        # Get the global features
        glofeat = grf[ix, 0:n_frames, :]

        # Get the local features by concatenating the features
        locfeat = np.zeros((batch_size, n_frames, n_partitions-1, n_features))

        k = 0
        for idx in ix:

            # Local features pool2
            featl2 = sio.loadmat(trainlist2[idx])
            featl2 = featl2['feat']
            # featl2 = np.transpose(featl2, (2, 0, 1))  # Comment this for previous feature

            # Local features pool4
            featl4 = sio.loadmat(trainlist4[idx])
            featl4 = featl4['feat']
            # featl4 = np.transpose(featl4, (2, 0, 1))  # Comment this for previous feature

            # Concatenate the features
            featl = np.concatenate([featl2, featl4],axis = 1)

            # Place on corresponding batch
            locfeat[k,:,:,:] = featl
            k = k + 1
        
        # Placeholder
        global_input = glofeat
        local_input = locfeat
        labels_input = raw_labels[ix, :, :]
        labels_ohe_input = raw_labels_ohe[ix, :, :]

        # print(global_input.shape)
        # print(local_input.shape)

        # Convert the input to device
        global_input = torch.from_numpy(global_input).to(cuda_device)
        local_input = torch.from_numpy(local_input).to(cuda_device)
        labels_input = torch.from_numpy(labels_input).to(cuda_device)
        labels_ohe_input = torch.from_numpy(labels_ohe_input).to(cuda_device)

        # Forward
        _, CRF_Loss_Batch, Loss = TrainingModel(global_input, local_input, labels_input, labels_ohe_input)

        # Backpropagate
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        
        # Loss placeholder
        epoch_loss.append(Loss)
        epoch_crf_loss.append(CRF_Loss_Batch)
    
    LR_Record = LR_Scheduler.get_last_lr()
    LR_Record = LR_Record[0]
    LR_Scheduler.step()

    # Print the Loss
    loss = torch.mean(torch.stack(epoch_loss), dim=0).item()
    crf_loss = torch.mean(torch.stack(epoch_crf_loss), dim=0).item()
    print("Epoch:", epoch+1, " done. Loss:", loss, " CRFloss: ", crf_loss)
    
    # Tensorboard
    writer.add_scalar('training_Loss', loss, epoch)
    writer.add_scalar('crf_Loss', crf_loss, epoch)
    writer.add_scalar('LR', LR_Record, epoch)

    # WanDB logging
    if wandb_flag:
        wandb.log({
            "epoch" : epoch,
            "training_Loss" : loss,
            "crf_Loss" : crf_loss,
            "LR" : LR_Record
        })
    
    # Evaluation and saving model
    if((epoch) % 10 == 0):
        
        # Save the Model
        model_out_path = model_dir + "/model_epoch_" + str(epoch) + ".pth"
        torch.save(TrainingModel, model_out_path)
        
        # Evaluation
        best_out_path = model_dir + "/best_model.pth"
        
        TrainingModel.eval()
        with torch.no_grad():
            evaluator = test_PRID_old.Evaluator(evaluation_path, TrainingModel, dataset_name, experiment_name, epoch, batch_size, n_features, n_class, n_frames, n_partitions, cuda_device)
            Test_R1, Test_R5, Test_R20 = evaluator.get_evaluation()

        # Tensorboard
        writer.add_scalar('test_r1', Test_R1, epoch)
        writer.add_scalar('test_r5', Test_R5, epoch)
        writer.add_scalar('test_r20', Test_R20, epoch)

        # WanDB logging
        if wandb_flag:
            wandb.log({
                "epoch" : epoch,
                "test_r1" : Test_R1,
                "test_r5" : Test_R5,
                "test_r20" : Test_R20
            })

        # Save the best model if test accuracy is increased
        if Test_R1 > best_eval:
            torch.save(TrainingModel, best_out_path)
            best_eval = Test_R1
        
        # Change to Training Mode
        TrainingModel.train()
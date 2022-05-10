import numpy as np
import torch

# from models.CRF import CRF
from module.CRF_SelfAttention import CRF_SelfAttention
from Model import Model

assert torch.cuda.is_available()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training parameters
batch_size = 1

# Network parameters
n_features = 512    # Number of features from feature extractor
n_hidden = 512      # Number of unit for self-attention
n_frames = 18       # Number of temporal frames
n_partitions = 7    # Number of partitions / spatial frames (4 Local Pooling + 2 Local Pooling + 1 Global Pooling)
n_head = 8          # Number of self-attention head
n_cluster = 38      # Number of cluster (for compatibility matrix only ?)

#### Mars Dataset
n_class = 3       # Number of classes in dataset / Train = 625

# Model creation
# SA_CRF_Model = CRF_SelfAttention(n_features, n_hidden, n_class, n_frames, n_partitions, n_head, n_cluster, device=torch.device("cuda:0")).to(device)
# print(SA_CRF_Model)

TrainingModel = Model(batch_size, n_features, n_hidden, n_class, n_frames, n_partitions, n_head, n_cluster, device)
print(TrainingModel)

# multiscale_embed = torch.zeros(10, 10, 10).to(device)
# labels = torch.zeros(10, 10).to(torch.bool).to(device)

# cluster_features, CRF_Loss, context = SA_CRF_Model(multiscale_embed, labels)
# print("cluster_features: {}\n CRF_Loss: {}\n context: {}".format(cluster_features, CRF_Loss, context))

# Forward
train_num = 1

global_input = np.zeros((batch_size, n_frames, n_features))
local_input = np.zeros((batch_size, n_frames, n_partitions-1, n_features))
labels_input = np.zeros((batch_size, n_frames, 1))
labels_ohe_input = np.zeros((batch_size, n_frames, n_class))

# Convert the input to device
global_input = torch.from_numpy(global_input).to(device)
local_input = torch.from_numpy(local_input).to(device)
labels_input = torch.from_numpy(labels_input).to(device).type(torch.LongTensor)
labels_ohe_input = torch.from_numpy(labels_ohe_input).to(device).type(torch.LongTensor)

print(global_input.shape)
print(local_input.shape)
print(labels_input.shape)
print(labels_ohe_input.shape)

xemg, CRF_Loss_Batch, Loss = TrainingModel(global_input, local_input, labels_input, labels_ohe_input)
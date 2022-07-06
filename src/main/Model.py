import torch
from torch import nn
from torch.autograd import Variable

from module.CRF_SelfAttention import CRF_SelfAttention
from module.pos_encoding_sin import PositionalEncodingSin

class Model(nn.Module):
    """
    The full Model of the CRF as RNN with Self-Attention
    
    """

    def __init__(self, batch_size, n_features, n_hidden, n_class, n_frames, n_partitions, n_head, n_cluster, device):
        """
        Build the model with the specified parameter
        
        Args:
            batch_size:         Batch size of the input
            n_features:         Number of features from the feature extractor
            n_hidden:           Number of unit for self-attention
            n_class:            Number of classes used in the dataset
            n_frames:           Number of temporal frames
            n_partitions:       The size of partitions (Local + Global)
            n_head:             The number of self-atttention heads
            n_cluster:          The cluster size
        """
        super(Model, self).__init__()
        
        # Parameter initialization
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.n_frames = n_frames
        self.n_partitions = n_partitions
        self.n_cluster = n_cluster
        self.device = device

        self.dropout = 0.8

        # Sinusoidal Positional_Encoding
        self.pos_encoding = PositionalEncodingSin(n_features)
        
        # Build the CRF-RNN with self-attention Module
        self.CRF_Module = CRF_SelfAttention(n_features, n_hidden, n_class, n_frames, n_partitions, n_head, n_cluster, device)
        self.CRF_Module.to(self.device)
        
        # Graph representation layer
        self.graph_linear = nn.Linear(n_features*2, n_hidden).to(self.device)
        self.graph_dropout = nn.Dropout(p=self.dropout).to(self.device)

        # Graph linear embedddings
        # self.weights = {
        #     'inp1' : nn.Parameter(torch.empty(n_features, n_hidden).normal_(mean=0.0, std=0.01)).to(self.device),
        #     'inp2' : nn.Parameter(torch.empty(n_features, n_hidden).normal_(mean=0.0, std=0.01)).to(self.device),
        #     'out1' : nn.Parameter(torch.empty(1*n_features, n_class).normal_(mean=0.0, std=0.01).to(self.device))
        # }

        # self.biases = {
        #     'inp1' : nn.Parameter(torch.empty(n_hidden).normal_(mean=0.0, std=0.01)).to(self.device),
        #     'inp2' : nn.Parameter(torch.empty(n_hidden).normal_(mean=0.0, std=0.01)).to(self.device),
        #     'out1' : nn.Parameter(torch.empty(n_class).normal_(mean=0.0, std=0.01).to(self.device))
        # }

        # Linear Classifier
        self.linear_classifier = nn.Linear(n_features, n_class).to(self.device)

        # self.weights = {
        #     'inp1' : Variable(torch.empty(n_features, n_hidden).normal_(mean=0.0, std=0.01), requires_grad=True).to(self.device),
        #     'inp2' : Variable(torch.empty(n_features, n_hidden).normal_(mean=0.0, std=0.01), requires_grad=True).to(self.device),
        #     'out1' : Variable(torch.empty(1*n_features, n_class).normal_(mean=0.0, std=0.01), requires_grad=True).to(self.device)
        # }
        
        # self.biases = {
        #     'inp1' : Variable(torch.empty(n_hidden).normal_(mean=0.0, std=0.01), requires_grad=True).to(self.device),
        #     'inp2' : Variable(torch.empty(n_hidden).normal_(mean=0.0, std=0.01), requires_grad=True).to(self.device),
        #     'out1' : Variable(torch.empty(n_class).normal_(mean=0.0, std=0.01), requires_grad=True).to(self.device)
        # }
        
    def forward(self, global_features, local_features, labels, labels_ohe):
        """
        Forward inference of the model
        
        Args:            
            local_features:      Local features from the feature extractor [batch_size, n_frame, n_partitions-1, n_features]
            global_features:     Global features from the feature extractor [batch_size, n_frame, , n_features]
            labels:              Ground truth labels
            labels_ohe:          Ground truth labels in OHE format
        """
        
        # Expand the global features dimension
        emb_global = global_features[:, :, None, :]

        # loss parameter
        CRF_Loss_Batch = 0.0
        Loss = 0.0

        # Positional Encoding
        position_idx = torch.ones(self.batch_size, self.n_frames, self.n_hidden)
        position_enc = self.pos_encoding(position_idx)
        position_enc = torch.reshape(position_enc, (1, self.n_frames, self.n_hidden))
        position_enc = torch.tile(position_enc, [self.n_partitions, 1, 1])
        position_enc = position_enc.permute(1, 0, 2).to(self.device)

        # Iterate through batch
        for i in range(0, self.batch_size):
            
            # Conconcatenate the global with local features and add positional encoding
            multiscale_embed = torch.cat((local_features[i,:,:,:], emb_global[i,:,:,:]), dim=1) + position_enc
            
            # Forward to self-attention with CRF layer
            cluster_features, CRF_Loss, sa_features = self.CRF_Module(multiscale_embed.float(), labels_ohe[i, 0, :])

            cluster_features = torch.sum(cluster_features, dim=0, keepdim=True)
            
            # Add the loss
            CRF_Loss_Batch = CRF_Loss_Batch + CRF_Loss
            
            # Append the cluster features 
            if i == 0:
                attention_list = cluster_features
            else:
                attention_list = torch.cat([attention_list, cluster_features], dim=0)
        
        # Linear embedding for graph representation
        xemg = torch.cat([global_features[0,:,:], torch.tile(attention_list, [self.n_frames, 1])], dim=1).float()
        xemg = self.graph_linear(xemg)
        xemg = self.graph_dropout(xemg)
        
        # Labels assignment
        y_data = labels_ohe[0,:,:]
        zc = torch.squeeze(torch.cat(torch.chunk(labels, self.n_frames, dim=1), dim=0))
        
        # Linear classifier
        
        # class_output = torch.matmul(xemg, self.weights['out1']) + self.biases['out1']
        class_output = self.linear_classifier(xemg)
        
        # print(class_output.shape)

        # Calculate graph loss
        CrossEntropyLoss = nn.CrossEntropyLoss()
        CELoss = (CrossEntropyLoss(class_output, y_data))
        CELoss = torch.mean(CELoss)
        
        CRF_Loss_Batch = CRF_Loss_Batch/self.batch_size
        
        Loss = CELoss + CRF_Loss_Batch
        
        return xemg, CRF_Loss_Batch, Loss
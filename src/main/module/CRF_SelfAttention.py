import torch
from torch import nn

class CRF_SelfAttention(nn.Module):
    """
    PyTorch implementation of the Self-Attention CRF-RNN Module
    
    """
    
    def __init__(self, n_features, n_hidden, n_class, n_frames, n_partitions, n_head, n_cluster, device="cuda:0"):
        """
        Build the CRF as RNN with Self-Attention Module
        
        Args:
            n_features:         Number of features from the feature extractor
            n_hidden:           Number of unit for self-attention
            n_class:            Number of classes used in the dataset
            n_frames:           Number of temporal frames
            n_partitions:       The size of partitions (Local + Global)
            n_head:             The number of self-atttention heads
            n_cluster:          The cluster size
            device:             The device
        Return:
            Object:             The CRF with self-attention module
        """
        super(CRF_SelfAttention, self).__init__()
        
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.n_frames = n_frames
        self.n_partitions = n_partitions
        self.n_head = n_head
        self.n_cluster = n_cluster

        self.device = device
        
        # Maximum number of iterations in CRF as RNN
        self.max_iterations = 2

        # Compatibility matrix > identity matrix with n_cluster x n_cluster
        self.compatibility_matrix = torch.eye(n=n_cluster,
                                              m=n_cluster,
                                              dtype=torch.float32).to(device)

        # Unary energy calculation layer
        self.unary = nn.Linear(self.n_features, self.n_cluster)
        
        # Halting probability layer
        self.halting_linear = nn.Linear(self.n_features, 1)
        self.halting_linear.bias.data.fill_(1.0) # Bias initialization

        ### Message Passing Layer

        # Temporal attention (multiple multihead self-attention layer) based on temporal locality
        self.multihead_attn_scale2 = nn.MultiheadAttention(embed_dim=self.n_hidden, num_heads=self.n_head, dropout=0.1)
        self.multihead_attn_scale4 = nn.MultiheadAttention(embed_dim=self.n_hidden, num_heads=self.n_head, dropout=0.1)
        self.multihead_attn_scale6 = nn.MultiheadAttention(embed_dim=self.n_hidden, num_heads=self.n_head, dropout=0.1)
        
        # Linear layer in message passing
        self.message_passing_linear = nn.Linear(self.n_features, self.n_cluster)

        # CLuster features linear layer
        self.cluster_features_linear = nn.Linear(self.n_features, self.n_class)

    # Not used for this version        
    def adjacency(self, adj=None):
        """
        Adjacency between the two partitions at all frames
        Adjancency Spatial-Temporal = [n_frames * n_size, n_frames * n_size]

        Args:            
            multiscale_embed:   Multiscale embedding from global and local features
            n_hidden:           Number of unit for self-attention
        """
        if not adj:
            adjacency = torch.ones((self.n_partitions*self.n_frames, self.n_partitions*self.n_frames))
        return adjacency
    
    def temporal_attention(self, embedding, adjacency):
        """
        Self-Attention on different scale of temporal locality
        
        Args:            
            embedding:    Embedding that used as query and key for the self-attention 
            adjacency:    Adjacency matrix of the graph
        """
        
        # Normalization constant
        norm_scale_2 = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=torch.float32).to(self.device)
        norm_scale_4 = torch.tensor([1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1], dtype=torch.float32).to(self.device)
        norm_scale_6 = torch.tensor([1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2, 1], dtype=torch.float32).to(self.device)
        
        norm_scale_2d = torch.tile(norm_scale_2[:, None, None], [1, self.n_partitions, self.n_hidden])
        norm_scale_4d = torch.tile(norm_scale_4[:, None, None], [1, self.n_partitions, self.n_hidden])
        norm_scale_6d = torch.tile(norm_scale_6[:, None, None], [1, self.n_partitions, self.n_hidden])
        
        # Initial value
        gaccum = torch.zeros_like(embedding).to(self.device)
        
        # Multiscale part features
        group1 = embedding

        ############# Temporal Locality Scale 2 #############
        scale = 2
        gval = torch.zeros([0, self.n_partitions, self.n_hidden]).to(self.device)
        
        for ind in range(0, (self.n_frames-scale+1)):
            
            # Create cliques
            st = ind
            et = ind + scale
            
            # Select the features from ind to ind + scale (temporal frame)
            group2 = group1[st:et, :, :]
            adj = adjacency[(st*self.n_partitions):(et*self.n_partitions), (st*self.n_partitions):(et*self.n_partitions)]
            
            # Need to confirm the chunk operation on rizard code
            soft_attention = torch.cat(torch.chunk(group2, scale, dim=0), dim=1)
            
            # Forward through multihead attention
            multi_g, _ = self.multihead_attn_scale2(soft_attention, soft_attention, soft_attention)
            
            multi_g = torch.cat(torch.chunk(multi_g, scale, dim=1), dim=0)
            
            if ind == 0:
                gval = torch.concat([gval, multi_g], dim=0) # scales, n_size, n_hidden
            else:
                gvalt = torch.concat([(torch.zeros([ind, self.n_partitions, self.n_hidden]).to(self.device)), multi_g], dim=0)
                gval = torch.concat([gval, (torch.zeros([1, self.n_partitions, self.n_hidden]).to(self.device))], dim=0)
                gval = gval + gvalt
        
        # Accumulate self-attention outputs
        gaccum = gaccum + torch.divide(gval, norm_scale_2d) 
        
        ############# Temporal Locality Scale 4 #############
        scale = 4
        gval = torch.zeros([0, self.n_partitions, self.n_hidden]).to(self.device)
        
        for ind in range(0, (self.n_frames-scale+1)):
            
            # Create cliques
            st = ind
            et = ind + scale
            
            # Select the features from ind to ind + scale (temporal frame)
            group2 = group1[st:et, :, :]
            adj = adjacency[(st*self.n_partitions):(et*self.n_partitions), (st*self.n_partitions):(et*self.n_partitions)]
            
            # Need to confirm the chunk operation on rizard code
            soft_attention = torch.cat(torch.chunk(group2, scale, dim=0), dim=1)
            
            # Forward through multihead attention
            multi_g, _ = self.multihead_attn_scale2(soft_attention, soft_attention, soft_attention)
            
            multi_g = torch.cat(torch.chunk(multi_g, scale, dim=1), dim=0)
            
            if ind == 0:
                gval = torch.concat([gval, multi_g], dim=0) # scales, n_size, n_hidden
            else:
                gvalt = torch.concat([(torch.zeros([ind, self.n_partitions, self.n_hidden]).to(self.device)), multi_g], dim=0)
                gval = torch.concat([gval, (torch.zeros([1, self.n_partitions, self.n_hidden]).to(self.device))], dim=0)
                gval = gval + gvalt
        
        # Accumulate self-attention outputs
        gaccum = gaccum + torch.divide(gval, norm_scale_4d)
        
        ############# Temporal Locality Scale 6 #############
        scale = 6
        gval = torch.zeros([0, self.n_partitions, self.n_hidden]).to(self.device)
        
        for ind in range(0, (self.n_frames-scale+1)):
            
            # Create cliques
            st = ind
            et = ind + scale
            
            # Select the features from ind to ind + scale (temporal frame)
            group2 = group1[st:et, :, :]
            adj = adjacency[(st*self.n_partitions):(et*self.n_partitions), (st*self.n_partitions):(et*self.n_partitions)]

            soft_attention = torch.cat(torch.chunk(group2, scale, dim=0), dim=1)
            
            # Forward through multihead attention
            multi_g, _ = self.multihead_attn_scale2(soft_attention, soft_attention, soft_attention)
            
            multi_g = torch.cat(torch.chunk(multi_g, scale, dim=1), dim=0)
            
            if ind == 0:
                gval = torch.concat([gval, multi_g], dim=0) # scales, n_size, n_hidden
            else:
                gvalt = torch.concat([(torch.zeros([ind, self.n_partitions, self.n_hidden]).to(self.device)), multi_g], dim=0)
                gval = torch.concat([gval, (torch.zeros([1, self.n_partitions, self.n_hidden]).to(self.device))], dim=0)
                gval = gval + gvalt
        
        # Accumulate self-attention outputs
        gaccum = gaccum + torch.divide(gval, norm_scale_6d)
        
        # Return the accumulation of self-attention at different temporal locality scales
        return 0.25*gaccum
    
    def compatibility_transform(self, pairwise):
        """
        Compatibility transform by matrix multiplication
        
        Args:            
            pairwise:   Pairwise energy [n_partitions, n_frames, n_cluster]
        Return:
            pairwise:   Pairwise energy [n_frames, n_partitions, n_cluster]
        """
        pairwise = torch.cat(torch.chunk(pairwise, self.n_partitions, 0), dim=1)
        pairwise = torch.matmul(self.compatibility_matrix, torch.unsqueeze(torch.squeeze(pairwise), dim=2))
        pairwise = torch.unsqueeze(torch.squeeze(pairwise), dim=0)
        pairwise = torch.cat(torch.chunk(pairwise, self.n_partitions, 1), dim=0)
        return pairwise
    
    def message_passing(self, multiscale_embed):
        """
        Perform calculation of pairwise energy using self-attention
        
        Args:            
            multiscale_embed:   Multiscale embedding from global and local features
        """
        
        # Find adjacency
        adjacency = self.adjacency()

        # !! For now the adjacency isn't used because the adjacency is
        # used inside the multihead self-attention so we need to change
        # the module

        # Temporal self-attention
        pairwise = self.temporal_attention(multiscale_embed, adjacency)
        pairwise = pairwise.permute(1, 0, 2)
        pairwise = self.message_passing_linear(pairwise)
        
        return pairwise
    
    def halting_probability(self, embed, ptn, Rt, Nt):
        """
        Calculate the halting probability for the mean-field inference loop

        Args:            
            embed:           Multiscale embedding from global and local features
            ptn, Rt, Nt:     Halting parameters
        Return:
            p:               Previous states
            ptn, Rt, Nt:     New halting parameters
            new_halted, run: Halting flag
        """
        
        ####
        p = self.halting_linear(embed)
        p = torch.sigmoid(p)
        p = torch.mean(p, dim=1)
        p = torch.squeeze(p)
        
        ####
        run = torch.less(ptn, 1.0).float()
        new_halted = torch.greater((ptn + p * run), 0.99).float() * run
        run = torch.le((ptn + p * run), 0.99).float() * run
        
        ####
        ptn += p * run
        Rt += new_halted * (1 - ptn)
        ptn += new_halted * Rt
        Nt += run + new_halted
        
        return p, ptn, Rt, Nt, new_halted, run
    
    def transition_function(self, embed, p, Rt, new_halted, run, prev):
        """
        Calculate the relational context

        Args:            
            embed:                  Multiscale embedding from global and local features
            p, Rt, new_halted, run: Halting parameters
            prev:                   Previous states
        Return:
            y:  Relational context
        """
        weights = torch.unsqueeze((p * run + new_halted * Rt), dim=-1)
        weights = torch.unsqueeze(weights, dim=-1)        
        weights = torch.tile(weights, [1, self.n_partitions, self.n_features])
        y = torch.multiply(embed, weights) + torch.multiply(prev , 1-weights)
        
        return y
    
    # TODO: Contrastive loss function
    # def contrastive_loss(self, marginal, context):
    #     """
    #     Calculate the relational context

    #     Args:            
    #         marginal:         Marginal distribution
    #         context:          Relational context
    #     Return:
    #         contrastive_loss: Contrastive loss of the features
    #     """

    #     return 0
    
    def calculate_cluster_features(self, marginal, context):
        """
        Compute the clusters features

        Args:            
            marginal:         Marginal distribution
            context:          Relational context
        Return:
            classes:          Classes features
            cluster_features: Cluster features
        """
        marginal = torch.nn.functional.softmax(marginal, dim=2)
        marginal = torch.squeeze(torch.cat(torch.chunk(marginal, self.n_partitions, dim=0), dim=1))
        contextc = torch.squeeze(torch.cat(torch.chunk(context, self.n_partitions, dim=1), dim=0))
        cluster_features = torch.matmul(torch.transpose(marginal, 1, 0), contextc)
        classes = self.cluster_features_linear(torch.sum(cluster_features, dim=0, keepdim=True))
        return classes, cluster_features

    def forward(self, multiscale_embed, labels):
        """
        Forward inference of the CRF as RNN with Self-Attention
        
        Args:            
            multiscale_embed:   Multiscale embedding from global and local features
            labels:             Labels of the data
        """
        
        # Calculate Unary Energy (Eu) for Initial Marginal Distribution (E)
        # Equation 2 & 3

        # Initialize with zeros
        marginal = torch.zeros(self.n_partitions, self.n_frames, self.n_cluster)

        # Context
        context = torch.zeros_like(multiscale_embed).to(self.device)
        
        # Input shape as list
        input_shape = list(multiscale_embed.shape)

        # Halting probability variable
        ptn = torch.zeros(input_shape[0:1]).to(self.device)
        Rt = torch.zeros(input_shape[0:1]).to(self.device)
        Nt = torch.zeros(input_shape[0:1]).to(self.device)
        
        # Calculate the initial marginal energy
        marginal = self.unary(multiscale_embed)
        marginal = marginal.permute(1, 0, 2)
        
        # Iteration step
        iter_step = 0
        
        #### Perform mean-field inference iteration
        # Inputs :
        #    multiscale features
        #    compatibility matrix
        #    marginal distribution
        
        # Perform mean-field inference iteration
        # Until max iteration reached or halting probability flag occured
        while True:
            # Compute the halting probability
            # Equation 4 / Equation 12 on Group Activity Paper
            p, ptn, Rt, Nt, new_halted, run = self.halting_probability(multiscale_embed, ptn, Rt, Nt)
            
            # Message Passing using Self-Attention
            # Equation 5
            pairwise = self.message_passing(multiscale_embed)
            
            # Compatibility Transform
            # Equation 6
            pairwise = self.compatibility_transform(pairwise)
            
            # Unary Addition = Unary + Pairwise
            # Equation 7
            marginal = -pairwise + marginal
            
            # Transition Function for updating the relational context
            context = self.transition_function(multiscale_embed, p, Rt, new_halted, run, context)
            
            # For the next iteration
            multiscale_embed = multiscale_embed.view(input_shape)
            context = context.view(input_shape)
            
            # Reshape
            for i in [ptn, Rt, Nt]:
                i = i.view(input_shape[0:1])
                
            iter_step += 1
            
            # Check if already reached max iteration or halting probability reach 1
            flag = torch.any(torch.logical_and(torch.less(ptn, 0.99), torch.less(Nt, self.max_iterations)))
            
            # Break if condition meets
            if(flag == False):
                break
        
        #### Calculation after mean-field inference
        
        # TODO: Calculate the contrastive loss
        # contrastive_loss = self.contrastive_loss(marginal, context)
        
        # Compute clusters features
        classes, cluster_features = self.calculate_cluster_features(marginal, context)
        
        # Calculate pondering loss
        act_loss = torch.sum(Nt+Rt)
        
        # Calculate CRF Loss (without contrastive loss)
        CrossEntropyLoss = nn.CrossEntropyLoss()
        CELoss = 0.1*(CrossEntropyLoss(classes, labels[None,]))
        
        # Combine the loss
        CRF_Loss =  CELoss + (0.001 * act_loss)
        
        return cluster_features, CRF_Loss, context
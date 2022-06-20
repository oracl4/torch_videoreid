import os
import glob
import torch
import numpy as np
import scipy.io as sio

import oct2py
from oct2py import octave
from tqdm import tqdm

from Model import Model
from module.CRF_SelfAttention import CRF_SelfAttention

class Evaluator():

    def __init__(self, evaluation_path, model, dataset_name, experiment_name, epoch, batch_size, n_features, n_class, n_frames, n_partitions, device):
        
        # CUDA Device
        self.device = device
        self.model = model
        self.dataset_name = dataset_name
        self.epoch = epoch
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_class = n_class
        self.n_frames = n_frames
        self.n_partitions = n_partitions
        
        # OLD
        # Global features path
        self.globalfeat_path = '../../features/input/LSVID/previous/test/test_glofeat.mat'
        self.localfeat2_path = '../../features/input/LSVID/previous/test/test_split_2/'
        self.localfeat4_path = '../../features/input/LSVID/previous/test/test_split_4/'
        
        # Output features path
        self.output_features_dir = '../../features/output/' + dataset_name + "/" + experiment_name + "/"
        self.output_features_path = self.output_features_dir + "epoch_" + str(epoch) + ".mat"

        # Create the directory if not exists
        if not os.path.exists(self.output_features_dir):
            os.makedirs(self.output_features_dir)

        # Evaluation path
        self.evaluation_dir = evaluation_path
    
    def get_evaluation(self):
        
        print("##########   Extracting Test Features")

        # Find the Test Set
        testlist2  = sorted(glob.glob(self.localfeat2_path+'/*.mat'))
        testlist4  = sorted(glob.glob(self.localfeat4_path+'/*.mat'))
        test_num  = len(testlist2)

        feat = np.zeros((test_num, self.n_features))

        # Load the whole global feature
        gsf = sio.loadmat(self.globalfeat_path)
        gsf = gsf['glofeat']

        for i in tqdm(range(0, test_num)):
            
            # Select the global feature
            glof = np.zeros((self.batch_size, self.n_frames, self.n_features))
            glof[0,:,:] = gsf[i, 0:self.n_frames, :]
            
            # Prepare the local feature
            locf = np.zeros((self.batch_size, self.n_frames, self.n_partitions-1, self.n_features))
            
            # Load the 2 partition local feature
            featl2 = sio.loadmat(testlist2[i])
            featl2 = featl2['feat']
            # featl2 = np.transpose(featl2, (2, 0, 1))    # Comment this for previous feature
            
            # Load the 4 partition local feature
            featl4 = sio.loadmat(testlist4[i])
            featl4 = featl4['feat']
            # featl4 = np.transpose(featl4, (2, 0, 1))    # Comment this for previous feature
            
            # Combine the local feature
            featl = np.concatenate([featl2, featl4],axis = 1)
            locf[0,:,:,:] = featl
            
            # Convert feature to torch tensor
            glof = torch.from_numpy(glof).to(self.device)
            locf = torch.from_numpy(locf).to(self.device)
            
            # Dummy label
            labels_input = torch.zeros(self.batch_size, self.n_frames, 1).to(self.device)
            labels_ohe_input = torch.zeros(self.batch_size, self.n_frames, self.n_class).to(self.device)
            
            # Feed forward to the model
            feature, CRF_Loss_Batch, Loss = self.model(glof, locf, labels_input, labels_ohe_input)
            
            # Insert the feature to test feature list
            feat[i,:] = feature[0,:].cpu().data.numpy()

        # # Save the output features
        sio.savemat(self.output_features_path, {"feat": feat}, do_compression=False)
        
        print("##########   Calling evaluation Function")

        # Call the octave function
        octave.addpath(self.evaluation_dir)
        octave.addpath(os.path.join(self.evaluation_dir, "data"))
        octave.eval("pkg load statistics")
        
        print("Performing evaluation on test features")
        print("Test features : ", self.output_features_path)

        map_, r1_precision = octave.eval_func(self.output_features_path, nout=2)
        print("##########")

        return map_, r1_precision
        
# Perform Evaluation
if __name__ == '__main__':
    
    # CUDA Device
    cuda_device = torch.device("cuda:1")

    # Network parameters
    batch_size = 1
    n_class = 1042       # Mars 625
    n_features = 512    # Number of features from feature extractor
    n_frames = 18       # Number of temporal frames
    n_partitions = 7    # Number of partitions / spatial frames (4 Local Pooling + 2 Local Pooling + 1 Global Pooling)

    ###### Extract feature and get evaluation
    
    # Load pre-trained model
    pretrained_path = "../../work_dir/LSVID/base_newFeatex/model/best_model.pth"
    TestModel = torch.load(pretrained_path, map_location=cuda_device)
    TestModel.to(cuda_device)
    TestModel.eval()
    
    # Create evaluator
    evaluator = Evaluator(TestModel, "LSVID", "new_featex", 0, batch_size, n_features, n_class, n_frames, n_partitions, cuda_device)

    # Extract the output features
    map_, r1_precision = evaluator.get_evaluation()
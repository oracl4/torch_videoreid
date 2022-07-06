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

    def __init__(self, eval_dir, model, dataset_name, experiment_name, epoch, batch_size, n_features, n_class, n_frames, n_partitions, device):
        
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
        
        # Features Path
        self.localfeat1_path = '../../features/input/PRID/hope/test/split_1/'
        self.localfeat2_path = '../../features/input/PRID/hope/test/split_2/'
        self.localfeat4_path = '../../features/input/PRID/hope/test/split_4//'

        # Output features path
        self.output_features_dir = '../../features/output/' + dataset_name + "/" + experiment_name + "/"
        self.output_features_path = self.output_features_dir + "epoch_" + str(epoch) + ".mat"

        # Create the directory if not exists
        if not os.path.exists(self.output_features_dir):
            os.makedirs(self.output_features_dir)

        # Evaluation path
        self.evaluation_dir = eval_dir
    
    def get_evaluation(self):

        print("##########   Extracting Test Features")

        # Find the Test Set
        testlist1  = sorted(glob.glob(self.localfeat1_path+'/*.mat'))
        testlist2  = sorted(glob.glob(self.localfeat2_path+'/*.mat'))
        testlist4  = sorted(glob.glob(self.localfeat4_path+'/*.mat'))
        test_num  = len(testlist1)

        feat = np.zeros((test_num, self.n_features))

        for i in tqdm(range(0, test_num)):
            
            # Prepare the global feature
            glof = sio.loadmat(testlist1[i])
            glof = glof['feat']
            glof = np.transpose(glof, (1, 0, 2))
            
            # Prepare the local feature
            locf = np.zeros((self.batch_size, self.n_frames, self.n_partitions-1, self.n_features))
            
            # Load the 2 partition local feature
            featl2 = sio.loadmat(testlist2[i])
            featl2 = featl2['feat']
            
            # Load the 4 partition local feature
            featl4 = sio.loadmat(testlist4[i])
            featl4 = featl4['feat']
            
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

        # Save the output features
        sio.savemat(self.output_features_path, {"feat": feat}, do_compression=False)
        
        print("##########   Calling evaluation Function")

        # Call the octave function
        octave.addpath(self.evaluation_dir)
        octave.addpath(os.path.join(self.evaluation_dir, "data"))
        octave.eval("pkg load statistics")
        
        print("Performing evaluation on test features")
        print("Test features : ", self.output_features_path)

        Test_R1, Test_R5, Test_R20 = octave.eval_func(self.output_features_path, nout=3)
        
        print("##########")

        return Test_R1, Test_R5, Test_R20

# Perform Evaluation
if __name__ == '__main__':
    
    # CUDA Device
    cuda_device = torch.device("cuda:1")

    # Evaluation path
    evaluation_path =  '../evaluation/PRID'

    # Network parameters
    batch_size = 1
    n_class = 89       # Mars 625
    n_features = 512    # Number of features from feature extractor
    n_frames = 18       # Number of temporal frames
    n_partitions = 7    # Number of partitions / spatial frames (4 Local Pooling + 2 Local Pooling + 1 Global Pooling)

    ###### Extract feature and get evaluation
    
    # Load pre-trained model
    pretrained_path = "../../work_dir/PRID/previous_Featex/model/best_model.pth"
    TestModel = torch.load(pretrained_path, map_location=cuda_device)
    TestModel.to(cuda_device)
    TestModel.eval()
    
    # Create evaluator
    evaluator = Evaluator(evaluation_path, TestModel, "LSVID", "new_featex", 13, batch_size, n_features, n_class, n_frames, n_partitions, cuda_device)

    # Extract the output features
    R1, R5, R20 = evaluator.get_evaluation()
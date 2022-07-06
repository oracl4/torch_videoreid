import os
import random, torch, dataset, cv2, time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import resnet_old as resnet
import scipy.io as sio
from tqdm import tqdm

# ===================================================================================================
# All parameter, change before extracting

# CUDA Device
cuda_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

# ## Mars Dataset
# dataset_name = "Mars"
# num_classes = 625
# image_dir = '../../dataset/Mars/images/'
# pretrained_model = '../../model/feature_extraction/pretrained/resnet50_mars.pth'
# sequence_file_list = ['../../dataset/Mars/seq_list/list_train_seq.txt', '../../dataset/Mars/seq_list/list_test_seq.txt']

## LS-VID Dataset
dataset_name = "LSVID"
num_classes = 842
image_dir = '../../dataset/LS-VID/tracklet/'
pretrained_model = '../../work_dir/featex/LSVID/hope/model/model_epoch_390.pth'
sequence_file_list = ['../../dataset/LS-VID/list_sequence/list_seq_train.txt', '../../dataset/LS-VID/list_sequence/list_seq_test.txt']

# ## iLIDS-VID Dataset
# dataset_name = "ILIDS"
# num_classes = 136
# image_dir = '../../dataset/iLIDS-VID/i-LIDS-VID/sequences/'
# pretrained_model = '../../work_dir/featex/ILIDS/hope/model/model_epoch_120.pth'
# sequence_file_list = ['../../dataset/iLIDS-VID/sequence_file/train001.txt', '../../dataset/iLIDS-VID/sequence_file/test001.txt']

## PRID Dataset
# dataset_name = "PRID"
# num_classes = 89
# image_dir = '../../dataset/prid/multi_shot/'
# pretrained_model = '../../work_dir/featex/PRID/hope/model/model_epoch_100.pth'
# sequence_file_list = ['../../dataset/prid/sequence_file/train001.txt', '../../dataset/prid/sequence_file/test001.txt']

batch_size = 1
temporal_frame = 18

split_list = [1, 2, 4]
experiment_name = "hope"

output_path = '../../features/input/'
# ===================================================================================================

print("CUDA       :", cuda_device)
print("Dataset    :", dataset_name)
print("Pretrained :", pretrained_model)
    
# Dataset functions
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([ transforms.ToTensor(),  normalizer, ])

# Iterate through train and test list
for sequence in sequence_file_list:

    if "train" in sequence:
        train_test = "train"
    if "test" in sequence:
        train_test = "test"

    # Iterate through all split
    for split in split_list:
        
        output_dir = os.path.join(output_path, dataset_name, experiment_name, train_test, ("split_" + str(split)))        

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Output Directory :", output_dir)
        print("Sequence File :", sequence)
        print("Split :", split)

        # Create dataset and dataloader
        dataset_ = dataset.videodataset(dataset_dir=image_dir,
                                        txt_path=sequence,
                                        new_height=256,
                                        new_width=128,
                                        frames=temporal_frame,
                                        transform=transform)

        dataloader_ = torch.utils.data.DataLoader(dataset=dataset_,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=1)

        # Load the pretrained model
        model, _ = resnet.resnet50(pretrained=pretrained_model,
                                    num_classes=num_classes,
                                    train=False,
                                    frames=temporal_frame,
                                    split=split,
                                    partition=1)    # Always start with first partition

        model.cuda()
        model.eval()

        # Get feature function
        def get_feature(model, images):
            
            with torch.no_grad():
                
                # Put the image to the device
                images = Variable(images).cuda()
                
                # Fix the dimension
                images = torch.transpose(images, 1, 2)
                images = images.view(images.size(0)*images.size(1), images.size(2), images.size(3), images.size(4))
                
                # Get the features
                output = model(images)
                feature = output[1].cpu().data.numpy()

                del images
                del output

            torch.cuda.empty_cache()

            return feature

        # Iterate through dataset to get all the feature
        index = 0
        for data in tqdm(dataloader_):
            
            index = index + batch_size

            # Get the images and label
            images, label = data

            # (Temporal Frames, Split, N Features)
            feature_placeholder = []

            # Iterate through the partition from 1 - n_split
            for i in range(1, split+1):
                
                # Change the partition
                _ = model.change_partition(i)

                # Get the feature
                feature = get_feature(model, images)
                feature = np.squeeze(feature)

                # print("feature shape:", feature.shape)

                feature_placeholder.append(feature)

            # Combine the Feature
            feature_out = np.stack(feature_placeholder)
            feature_out = np.transpose(feature_out, (2, 0, 1))
            
            # print(feature_out.shape)

            savepath = output_dir + '/' + '{0:05d}'.format(index)
            savepath = savepath + ".mat"
            
            # Save the feature
            sio.savemat(savepath, {"feat": feature_out})
import os
import random, torch, dataset, cv2, time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import resnet_ as resnet
import scipy.io as sio
from tqdm import tqdm

# ===================================================================================================
# All parameter, change before extracting

# CUDA Device
cuda_device = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

## Mars Dataset
dataset_name = "Mars"
num_classes = 1261
image_dir = '../../dataset/Mars/images/'
pretrained_model = '../../model/feature_extraction/pretrained/resnet50_mars.pth'
# sequence_file = '../../dataset/Mars/seq_list/list_train_seq.txt'    # Train List
sequence_file = '../../dataset/Mars/seq_list/list_test_seq.txt'     # Test List

batch_size = 1
partition = 1
experiment_name = "base"
split = "test"
output_path = '../../feature/input/'
output_dir = os.path.join(output_path, dataset_name, experiment_name, split, ("partition_" + str(partition)))

# ===================================================================================================

print("CUDA :", cuda_device)
print("Dataset :", dataset_name)
print("Sequence File :", sequence_file)
print("Output Directory :", output_dir)
print("Partition :", partition)

# Create output directory
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
    
# Dataset functions
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([ transforms.ToTensor(),  normalizer, ])

# Create dataset and dataloader
dataset_ = dataset.videodataset(dataset_dir=image_dir,
                                txt_path=sequence_file,
                                new_height=256,
                                new_width=128,
                                frames=18,
                                transform=transform)

dataloader_ = torch.utils.data.DataLoader(dataset=dataset_,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=1)

# Load the pretrained model
model, _ = resnet.resnet50(pretrained=pretrained_model, num_classes=num_classes, train=False, partition=partition)
model.cuda()
model.eval()

# Get feature function
def get_feature(model, images):
    with torch.no_grad():
        images = Variable(images).cuda()
        images = torch.transpose(images, 1, 2)
        images = images.view(images.size(0)*images.size(1), images.size(2), images.size(3), images.size(4))

        output = model(images)
        del images
        feature = output[1].cpu().data.numpy()		
        del output

    torch.cuda.empty_cache()
    
    return feature
    
# Iterate through dataset to get all the feature
index = 0
for data in tqdm(dataloader_):
    
    index = index + batch_size
    images, label = data

    feature = get_feature(model,images)
    feature = np.squeeze(feature)

    savepath = output_dir + '/' + '{0:05d}'.format(index)
    savepath = savepath + ".mat"
    
    # Save the feature
    sio.savemat(savepath, {"feat": feature})
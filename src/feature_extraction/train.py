import os, random, torch, cv2, time, dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import resnet_ as resnet
import random

import wandb
from tqdm import tqdm

# Module
from Triplet import *
from samplers import RandomIdentitySampler
import data_manager # random identity

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ===================================================================================================
# All parameter, change before starting

# WanDB logging
wandb_flag = True

#### Hyperparameter
base_lr = 0.0005
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1

num_epoches = 501
step_size = 120
batch_size = 16
warmup_iter = 10

# Augmentation
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),  normalizer,])

### Dataset Parameter

# Pretrained ResNet-50
resnet50_path = "../../model/feature_extraction/resnet50-19c8e357.pth"

#### LS-VID Dataset
dataset_name = "LSVID"
image_dir = '../../dataset/LS-VID/tracklet/'
sequence_file = '../../dataset/LS-VID/list_sequence/list_seq_train.txt'
num_classes= 3772

# Global Settings
partition = 1
experiment_name = "experiment"
split = "train"

# Work directory
work_dir = '../../work_dir/featex/'
work_dir = os.path.join(work_dir, dataset_name, experiment_name)

# Model and tensorboard dir
model_dir = os.path.join(work_dir, "model")
tensorboard_dir = os.path.join(work_dir, "tensorboard")

# ===================================================================================================

print("Dataset :", dataset_name)
print("Sequence File :", sequence_file)
print("Work Directory :", work_dir)
print("Model Directory :", model_dir)
print("Tensorboard Directory :", tensorboard_dir)
print("Partition :", partition)

### Initialization

# Create output directory
# Create the directory
if not os.path.exists(work_dir):
	os.makedirs(work_dir)

if not os.path.exists(model_dir):
	os.makedirs(model_dir)

if not os.path.exists(tensorboard_dir):
	os.makedirs(tensorboard_dir)

# Create dataset and dataloader
train_dataset = dataset.videodataset(dataset_dir=image_dir,
									txt_path=sequence_file,
									new_height=256,
									new_width=128,
									frames=18,
									transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											batch_size=batch_size,
											shuffle=True,
											num_workers=4,
											drop_last=True)
# Load the Model
model, new_param = resnet.resnet50(pretrained=resnet50_path, num_classes=num_classes, train=True, partition=1)
model = nn.DataParallel(model, device_ids=[0,1])
model.cuda()

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()

param = []
params_dict = dict(model.named_parameters())
for key, v in params_dict.items():
		param += [{ 'params':v,  'lr_mult':1}]

optimizer = torch.optim.SGD(param, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

# ===================================================================================================
# Function

# Learning rate adjustment
def adjust_lr(epoch, alpha):

	warmup_factor = (1.0/3.0) * (1 - alpha) + alpha

	if epoch < warmup_iter:
		lr = base_lr * warmup_factor * (gamma ** (epoch // step_size)) 
	else:
		lr = base_lr * (gamma ** (epoch // step_size))

	for g in optimizer.param_groups:
		g['lr'] = lr * g.get('lr_mult', 1)
	return lr


### Logging

# WanDB 
if wandb_flag:

    # Initialize the WanDB
    wandb.init(project="person-reid", entity="oracl4")

    # Save the parameter
    wandb.config = {
        "dataset_name": dataset_name,
        "experiment" : experiment_name,
        "base_lr" : base_lr,
        "epochs" : num_epoches,
        "batch_size" : batch_size
    }

# Tensorboard
writer = SummaryWriter(tensorboard_dir)

# ===================================================================================================
# Training Loop

best_acc = 0.0

print("====== Start Training ======")

# Loop through epoch
for epoch in range(0, num_epoches):
	
	# Adjust the learning rate
	alpha = epoch/(warmup_iter*1.0)
	lr = adjust_lr(epoch, alpha)

	# print(alpha)
	# print('-' * 10)
	# print('epoch {}'.format(epoch + 1))

	running_loss = 0.0
	running_acc = 0.0
	
	start = time.time()
	since = time.time()
	
	model.train()
	
	# Iterate through data in dataloader
	for i, data in tqdm(enumerate(train_loader, 1)):
		
		# Load the data
		images, label = data
		images = torch.transpose(images, 1, 2)
		images = Variable(images).cuda()
		images = images.reshape(images.size(0)*images.size(1), images.size(2), images.size(3), images.size(4))

		# Get the label
		label = Variable(label).cuda()
        
        # Perform random erasing augmentation on the image
		w = random.randint(1, 32)
		h = random.randint(1, 64)
		cw = random.randint(1, 127) 
		ch = random.randint(1, 255)
		
		stw = max(cw-w, 0)
		sth = max(ch-h, 0)
		etw = min(cw+w, 127)
		eth = min(ch+h, 255)
		pro = random.randint(0,10)

		if etw-stw > 0 or sth-eth > 0 and pro>5:
			imp = torch.rand(images.size(0), images.size(1),eth-sth,etw-stw)
			images[:,:,sth:eth,stw:etw] = imp

        # Forward the image through model
		out,_ = model(images)

		# print(out.shape)
		# print(label.shape)
		
		# Calculate the loss with CrossEntropy
		loss =  criterion(out, label)
		running_loss += loss.item() * label.size(0)
		_, pred = torch.max(out, 1)
		
		num_correct = (pred == label).sum()
		running_acc += num_correct.item()
		
		# Backpropagate
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# Print loss and accuracy for each n iteration
		# if i % 200 == 0:
		# 	print('[{}/{}] iter: {}/{}. lr: {} . Loss: {:.6f}, Acc: {:.6f} time:{:.1f} s'.format(epoch+1, num_epoches, i, len(train_loader), lr, running_loss/(batch_size*i), running_acc/(batch_size*i), time.time() - since))
		# 	since = time.time()
	
	# Print the epoch result
	print('[{}/{}] iter: {}/{}. lr: {} . Loss: {:.6f}, Acc: {:.6f}'.format(epoch+1, num_epoches, i, len(train_loader), lr, running_loss/(batch_size*i), running_acc/(batch_size*i)))
	print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch+1, running_loss/(len(train_dataset)), running_acc/(len(train_dataset))))
	print('Time:{:.1f} s'.format(time.time() - start))

	# Log the result
	epoch_loss = running_loss/(len(train_dataset))
	epoch_acc = running_acc/(len(train_dataset))
	
	# Tensorboard
	writer.add_scalar('loss', epoch_loss, epoch)
	writer.add_scalar('accuracy', epoch_acc, epoch)
	writer.add_scalar('LR', lr, epoch)

	# WanDB
	if wandb_flag:
		wandb.log({
			"epoch" : epoch,
			"loss" : epoch_loss,
			"accuracy" : epoch_acc,
			"LR" : lr
		})

	# For each n epoch save model
	if((epoch) % 50 == 0):

		# Save the model
		model_out_path = model_dir + "/model_epoch_" + str(epoch) + ".pth"
		torch.save(model.state_dict(), model_out_path)

		# Check the best model
		best_out_path = model_dir + "/best_model.pth"

		if epoch_acc > best_acc:
			torch.save(model.state_dict(), best_out_path)
			best_acc = epoch_acc
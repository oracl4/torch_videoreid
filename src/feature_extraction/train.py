import os, random, torch, cv2, time, dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import resnet_old as resnet
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
base_lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1

num_epoches = 400
step_size = 100
batch_size = 10
num_worker = 4
warmup_iter = 10

# Augmentation
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),  normalizer,])

### Dataset Parameter

# Pretrained ResNet-50
resnet50_path = "../../model/feature_extraction/resnet50-19c8e357.pth"

#### Mars
dataset_name = "Mars"
image_dir = '../../dataset/Mars/images/'
sequence_file = '../../dataset/Mars/seq_list/list_train_seq.txt'
num_classes= 625

#### LS-VID Dataset
# dataset_name = "LSVID"
# image_dir = '../../dataset/LS-VID/tracklet/'
# sequence_file = '../../dataset/LS-VID/list_sequence/list_seq_train.txt'
# num_classes= 842

#### PRID Dataset
# dataset_name = "PRID"
# image_dir = '../../dataset/prid/multi_shot/'
# sequence_file = '../../dataset/prid/sequence_file/train001.txt'
# num_classes = 89

#### ILIDS Dataset
# dataset_name = "ILIDS"
# image_dir = '../../dataset/iLIDS-VID/i-LIDS-VID/sequences/'
# sequence_file = '../../dataset/iLIDS-VID/sequence_file/train001.txt'
# num_classes= 136

# Global Settings
split=1	
partition = 1
temporal_frame = 16
experiment_name = "hope"

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
print("Split :", split)

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
									frames=temporal_frame,
									transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											batch_size=batch_size,
											shuffle=True,
											num_workers=num_worker,
											drop_last=True)
# Load the Model
model, new_param = resnet.resnet50(pretrained=resnet50_path,
									num_classes=num_classes,
									train=True,
									frames=temporal_frame,
									split=1,
									partition=1)

model = nn.DataParallel(model, device_ids=[0,1])
model.cuda()

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
criterion     = nn.CrossEntropyLoss()
criterion_cls = CrossEntropyLabelSmooth(num_classes).cuda()
criterion_tri = TripletLoss(margin=0.3)

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

#### Logging

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
		
		# Get the image and label
		images, label = data
		
		# Fix the temporal frame shape
		images = torch.transpose(images, 1, 2)

		# Put the image and label to tensor on GPU
		images = Variable(images).cuda()
		label = Variable(label).cuda()
		images = images.reshape(images.size(0)*images.size(1), images.size(2), images.size(3), images.size(4))

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
			imp = torch.rand(images.size(0), images.size(1), eth-sth, etw-stw)
			images[:,:,sth:eth,stw:etw] = imp

        # Forward the image through model
		out,_ = model(images)

		# print("Debugging")
		# print(out.shape)	
		# print(label.shape)

		# Calculate the loss with CrossEntropy
		loss =  criterion_cls(out, label)
		running_loss += loss.item() * label.size(0)
		_, pred = torch.max(out, 1)
		
		# Terminate
		# import sys
		# sys.exit()

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
	if((epoch) % 10 == 0):

		# Save the model
		model_out_path = model_dir + "/model_epoch_" + str(epoch) + ".pth"
		torch.save(model.state_dict(), model_out_path)

		# Check the best model
		best_out_path = model_dir + "/best_model.pth"

		if epoch_acc > best_acc:
			torch.save(model.state_dict(), best_out_path)
			best_acc = epoch_acc
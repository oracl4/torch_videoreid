import torch.nn as nn
import math, torch
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from NonLocalBlock1D import NonLocalBlock1D
from collections import OrderedDict

#from models_utils.rga_modules import RGA_Module

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000, train=True,spa_on=True, cha_on=True, s_ratio=8, c_ratio=8, d_ratio=8,height=256, width=128, partition=1):

		self.inplanes = 64
		
		super(ResNet, self).__init__()
		
		self.istrain = train
		self.frames = 18
		self.partition = partition

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

		# RGA Modules
		# self.rga_att1 = RGA_Module(256, (height//4)*(width//4), use_spatial=spa_on, use_channel=cha_on,
		# 						cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		# self.rga_att2 = RGA_Module(512, (height//8)*(width//8), use_spatial=spa_on, use_channel=cha_on,
		# 						cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		# self.rga_att3 = RGA_Module(1024, (height//16)*(width//16), use_spatial=spa_on, use_channel=cha_on,
		# 						cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		# self.rga_att4 = RGA_Module(2048, (height//16)*(width//16), use_spatial=spa_on, use_channel=cha_on,
		# 						cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		
		# self.maxpool = nn.MaxPool2d((2,2), stride=1)
		
		# Partition Pooling Layer
		if self.partition == 1:
			self.avgpool = nn.AvgPool2d((16,8)) # 1 part
		if self.partition == 2:
			self.avgpool = nn.AvgPool2d((8,8)) # 2 part
		if self.partition == 4:
			self.avgpool = nn.AvgPool2d((4,8)) # 4 part

		self.num_features = 128
		self.feat = nn.Linear(512 * block.expansion, self.num_features)

		#self.feat_bn = nn.BatchNorm1d(self.num_features*4)
		
		self.feat_bn = nn.BatchNorm1d(self.num_features*4)
		self.drop = nn.Dropout(0.5)
		self.classifier = nn.Linear(4*self.num_features, num_classes)

		self.Nonlocal_block0 = NonLocalBlock1D(128*4)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, std=0.001)
				init.constant_(m.bias, 0)

		init.kaiming_normal_(self.feat.weight, mode='fan_out')
		init.constant_(self.feat.bias, 0)

		self.feat1= nn.Conv2d(1, 128, kernel_size=(3,128), stride=1, dilation=(1,1), padding=(1,0), bias=False)
		self.feat2 = nn.Conv2d(1, 128, kernel_size=(3,128), stride=1, dilation=(2,1), padding=(2,0), bias=False)
		self.feat3 = nn.Conv2d(1, 128, kernel_size=(3,128), stride=1, dilation=(3,1), padding=(3,0), bias=False)
		
		init.normal_(self.feat1.weight, std=0.001)
		init.normal_(self.feat2.weight, std=0.001)
		init.normal_(self.feat3.weight, std=0.001)

		self.Nonlocal_block0 = NonLocalBlock1D(128*4)
		
		# self.Nonlocal_block1 = NonLocalBlock1D(128)
		# self.Nonlocal_block2 = NonLocalBlock1D(128)
		# self.Nonlocal_block3 = NonLocalBlock1D(128)


	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
							kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		# x = self.rga_att1(x)

		x = self.layer2(x)
		# x = self.rga_att2(x)
		
		x = self.layer3(x)
		# x = self.rga_att3(x)

		# Global (take all)
		x = self.layer4(x)

		print(x.shape)
		
		# Change the partition split
		n_split = self.partition
		
		for i in range(n_split):

			# Start-End the Split
			start = int(i * (16/n_split))
			end = int(start + (16/n_split))
			
			# print(start, end)

			x_ = x[:,:,start:end,:]

			x_ = self.avgpool(x_)

			x_ = x_.view(x.size(0), -1)
			x_ = self.feat(x_)

			# print(x_.shape)
			
			# x = x.unsqueeze(dim=0)
			x_ = x_.view(int(x_.size(0)/self.frames), self.frames, -1)

			# print(x_.shape)

			x0 = torch.transpose(x_, 1, 2)
			
			x_ = x_.unsqueeze(dim=1)

			x1 = self.feat1(x_).squeeze(dim=3)
			x2 = self.feat2(x_).squeeze(dim=3)
			x3 = self.feat3(x_).squeeze(dim=3)

			# print(x0.size(), x1.size(), x2.size(), x3.size())
			
			x_ = torch.cat((x0, x1, x2, x3), dim=1)

			x4 = self.Nonlocal_block0(x_)
			
			x_  = x4.mean(dim=2)

			if i == 0:				
				x_combined = x_
				x4_combined = x4
			else:
				x_combined = torch.cat((x_combined, x_))
				x4_combined = torch.cat((x4_combined, x4))

		# print(x_combined.shape)
		# print(x4_combined.shape)
		
		if self.istrain:
			x_combined = self.feat_bn(x_combined)
			x_combined = self.relu(x_combined)
			x_combined = self.drop(x_combined)
			x_combined = self.classifier(x_combined)

		return x_combined, x4_combined
		
def resnet50(pretrained='True', num_classes=1000, train=True, partition=1):
	
	model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, train, partition=partition)

	# if pretrained:
	# 	model.load_state_dict('resnet50-19c8e357.pth')

	weight = torch.load(pretrained)
	static = model.state_dict()

	# new_state_dict = OrderedDict()
	# for k, v in weight.items():
	# 	name = k[7:] # remove module.
	# 	new_state_dict[name] = v

	# model.load_state_dict(new_state_dict)

	# 1. filter out unnecessary keys
	weight = {k: v for k, v in weight.items() if k in static}
	# weight.pop('classifier.weight')  # uncomment when training
	# weight.pop('classifier.bias') 
	# 2. overwrite entries in the existing state dict
	#static.update(weight) 			
	model.load_state_dict(weight,strict=False)

	for name, param in weight.items():
		if name not in static:
			# print('not load weight ', name)
			continue
		if isinstance(param, nn.Parameter):
			# print('load weight ', name, type(param))
			param = param.data
			static[name].copy_(param)

	# #model.load_state_dict(weight)

	new_param = []
	for name, param in static.items():
		if name not in weight:
			# print('new param ', name)
			new_param.append(name)
	
	return model, new_param
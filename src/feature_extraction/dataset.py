import os, torch, random, cv2, glob
import numpy as np

from torch.utils import data

class videodataset(data.Dataset):

	def __init__(self, dataset_dir, txt_path, new_height, new_width, frames, transform):
		self.new_height = new_height
		self.new_width = new_width
		self.frames = frames
		self.dataset_dir = dataset_dir
		self.transform = transform

		with open(txt_path) as f:
			line = f.readlines()
			self.img_list = [os.path.join(dataset_dir, i.split()[0])+'*' for i in line]
			self.label_list = [int(i.split()[1]) for i in line]

	def __getitem__(self, index):
		im_dir = self.img_list[index]
		
		# print(im_dir)
		image_list = glob.glob(im_dir) # ilids, mars
		#image_list = glob.glob(im_dir+"/*.png") # prid
		#print(image_list)
		
		image_list.sort()
		
		images = []
		for name in image_list:
			end = name[-3:]
			if end in ['png', 'jpg', 'jpeg']:
				images.append(name)
		image_list = images


		im_paths = []   
		if len(image_list) >= self.frames:
			rand_frame = int(random.random()*(len(image_list)-self.frames))
			for i in range(rand_frame,rand_frame+self.frames):
				im_path = image_list[i]
				im_paths.append(im_path)
		else:
			for i in range(0,self.frames):
				im_path = image_list[i%len(image_list)]
				im_paths.append(im_path)

		# print(im_paths)
		
		frames = []

		for im_path in im_paths:
			image = cv2.imread(im_path)

			# pro = random.randint(0,10)
			# if pro>=5:
			# 	image = cv2.flip(image,1)

			image = cv2.resize(image,(self.new_width, self.new_height))
			image = image[:,:, ::-1]
			image = self.transform(image.copy())

			# print(image.size())
			frames.append(image.numpy())

		frames = np.array(frames, np.float32)
		#print(np.shape(frames))
		frames = np.transpose(frames, (1, 0, 2, 3))
		frames = torch.from_numpy(frames).float()
		#print(type(frames), frames.size())

		label = self.label_list[index]
		return frames, label

	def __len__(self):
		return len(self.label_list)
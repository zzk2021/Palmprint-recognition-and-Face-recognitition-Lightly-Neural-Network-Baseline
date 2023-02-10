from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data

class CK(data.Dataset):
	'''

	Args:
		train (bool, optional): If True, creates dataset from training set, otherwise
			creates from test set.
		transform (callable, optional): A function/transforms that takes in an PIL image
			and returns a transformed version.
		there are 135, 177, 75, 207, 84, 249, 54 images in data;
		we choose 123, 159, 66, 186, 75, 225, 48 images for training;
		we choose 12, 8, 9, 21, 9, 24, 6 images for testing;
		the split are in order according to the fold number.

	'''

	def __init__(self, split='Training', fold=1, transform=None):
		self.transform = transform
		self.split = split # training set or test set
		self.fold = fold # the k-fold cross validation
		self.data = h5py.File('./H5File/CK+.h5','r',driver='core')

		number = len(self.data['data_label']) # 981
		sum_number = [0, 135, 312, 387, 594, 678, 927, 981] # the sum of class number
		test_number = [12, 18, 9, 21, 9, 24, 6] # the number of each class

		test_index = []
		train_index = []

		for j in range(len(test_number)):
			for k in range(test_number[j]):
				if self.fold != 10: #the last fold start from the last element
					test_index.append(sum_number[j]+(self.fold-1)*test_number[j]+k)
				else:
					test_index.append(sum_number[j+1]-1-k)

		for i in range(number):
			if i not in test_index:
				train_index.append(i)


		#now load the picked numpy arrays
		if self.split == 'Training':
			self.train_data = []
			self.train_labels = []
			for ind in range(len(train_index)):
				self.train_data.append(self.data['data_pixel'][train_index[ind]])
				self.train_labels.append(self.data['data_label'][train_index[ind]])


		elif self.split == 'Testing':
			self.test_data = []
			self.test_labels = []
			for ind in range(len(test_index)):
				self.test_data.append(self.data['data_pixel'][test_index[ind]])
				self.test_labels.append(self.data['data_label'][test_index[ind]])


	def __getitem__(self, index):
		if self.split == 'Training':
			img, target = self.train_data[index], self.train_labels[index]
		elif self.split == 'Testing':
			img, target = self.test_data[index], self.test_labels[index]
		img = img[:,:,np.newaxis]
		img = np.concatenate((img,img,img),axis=2)
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, target

	def __len__(self):
		if self.split == 'Training':
			return len(self.train_data)
		elif self.split == 'Testing':
			return len(self.test_data)

if __name__ == '__main__':
	import transforms as transforms
	transform_train = transforms.Compose([
		transforms.ToTensor(),
	])
	# transform_train = None
	data = CK(split = 'Training', transform = transform_train)
	for i in range(3):
		print(data.__getitem__(i))
	print(data.__len__())

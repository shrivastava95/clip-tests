import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
from PIL import Image
import clip

cwd = os.getcwd()
module2add = '/'.join(cwd.split("/")[:-1])
sys.path.append(module2add)

from utils.tinyimagenet_utils import *


class TinyImageNetDataset(Dataset):
	def __init__(self, split, mode, transform):
		self.split = split
		self.transform = transform
		
		helper_path, classes, ids = get_split_data(self.split)
		print(helper_path)
		self.data = torch.load(helper_path)
		# print(self.data)
		self.classes = [c.lower() for c in classes]
		self.ids = ids
		self.class_prompt_path = get_class_prompt_features(self.classes, mode)
	
	def __len__(self):
		return len(self.data.keys())

	def __getitem__(self, idx):
		keys = list(self.data.keys())
		k = keys[idx]
		image_path = k[3:]
		# print(k)
		# print(image_path)
		image = Image.open(image_path).convert("RGB")

		if self.transform is not None:
			image = self.transform(image)
				
		label = int(self.data[k])
		return image, label

def testing():
	model, preprocess = clip.load("ViT-B/16", "cuda")
	train_dataset = TinyImageNetDataset("val", preprocess)
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

	images, labels = next(iter(train_loader))
	print(images.shape, labels.shape)	


if __name__ == "__main__":
	testing()
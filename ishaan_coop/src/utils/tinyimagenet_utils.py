from .output_methods import *
import torch
import os
from tqdm import tqdm
import clip


def id2class_mapping():
	save_path = "../../prompting/data/tinyimagenet/id2class_mapping.pt"	
	if os.path.exists(save_path):
		return save_path

	else:
		anno_file = "../../prompting/data/tinyimagenet/words.txt"
		mapping = {}

		with open(anno_file) as f:
			for line in tqdm(f.readlines()):
				line = line.strip()
				id_code = line[:9]
				class_name = line[10:].lower()
				
				if "," in class_name:
					class_name = " or ".join(class_name.split(",")[:3])

				mapping[id_code] = class_name

		torch.save(mapping, save_path)
		return save_path

def train_image_class():
	train_data_folder = "../../prompting/data/tinyimagenet/train"
	mapping_path = id2class_mapping()	
	mapping = torch.load(mapping_path)

	train_ids = os.listdir(train_data_folder)
	train_ids.sort()
	train_classes = [mapping[i] for i in train_ids]

	save_path = "../data/tinyimagenet/train_dataset_helper.pt"
	if os.path.exists(save_path):
		return save_path, train_classes, train_ids
	
	else:
		dataset_mapping = {}
		train_data_folder = "../../prompting/data/tinyimagenet/train"
		train_subfolders = 	[os.path.join(train_data_folder, i) for i in train_ids]
		
		for i, subfolder in enumerate(train_subfolders):
			image_subfolder = os.path.join(subfolder, "images")
			for fname in os.listdir(image_subfolder):
				image_path = os.path.join(image_subfolder, fname)
				dataset_mapping[image_path] = i
	
		torch.save(dataset_mapping, save_path)
		return save_path, train_classes, train_ids

def val_image_class():
	_, classes, ids = train_image_class()
	save_path = "../data/tinyimagenet/val_dataset_helper.pt"

	if os.path.exists(save_path):
		return save_path, classes, ids

	else:
		mapping_path = "../../prompting/data/tinyimagenet/val_image_ids.pt"	
		mapping = torch.load(mapping_path)
		dataset_mapping = {}

		for k, v in mapping.items():
			image_path = k.replace("tiny-imagenet-200/", "")
			idx = ids.index(v)
			dataset_mapping[image_path] = idx
		
		torch.save(dataset_mapping, save_path)
		return save_path, classes, ids

def get_split_data(split):
	if split == "train":
		return train_image_class()
	elif split == "val":
		return val_image_class()

def basic_prompt_features(classes):
	save_path = "../../prompting/data/tinyimagenet/basic_class_prompt_features.pt"
	if os.path.exists(save_path):
		return save_path

	else:
		model, _ = clip.load("ViT-B/16", "cuda")
		template = [f"a photo of {c.lower()}" for c in classes]
		texts = clip.tokenize(template).cuda()

		class_feats = []
		with torch.no_grad():
			for i in tqdm(range(texts.shape[0])):
				text_features = model.encode_text(texts[i].unsqueeze(0)).detach().cpu()
				class_feats.append(text_features.view((1, 512)))
		
		class_feats = torch.cat(class_feats, dim=0).view((len(classes), 512))
		torch.save(class_feats, save_path)
		return save_path	
	
def extended_prompt_features(classes, attributes):
	save_path = "../../prompting/data/tinyimagenet/extended_class_prompt_features.pt"
	if os.path.exists(save_path):
		return save_path

	else:
		model, _ = clip.load("ViT-B/16", "cuda")
		template = []
		for c in classes:
			prompt = f"a photo of {c} which has "
			for attr in attributes[c]:
				prompt += attr + ", "
			prompt = prompt[:-1] + "."
			template.append(prompt)
		
		texts = clip.tokenize(template, truncate=True).cuda()
	
		class_feats = []
		with torch.no_grad():
			for i in tqdm(range(texts.shape[0])):
				text_features = model.encode_text(texts[i].unsqueeze(0)).detach().cpu()
				class_feats.append(text_features.view((1, 512)))
		
		class_feats = torch.cat(class_feats, dim=0).view((len(classes), 512))
		torch.save(class_feats, save_path)
		return save_path	

def avg_token_prompt_features(classes):
	save_path = "../../prompting/data/tinyimagenet/avg_token_class_prompt_features.pt"
	if os.path.exists(save_path):
		return save_path

	else:
		model, _ = clip.load("ViT-B/16", "cuda")
		template = [f"a photo of {c.lower()}" for c in classes]
		texts = clip.tokenize(template).cuda()

		class_feats = []
		with torch.no_grad():
			for i in tqdm(range(texts.shape[0])):
				text_features = encode_text_with_token_avg(model, texts[i].unsqueeze(0)).detach().cpu()
				class_feats.append(text_features.view((1, 512)))
		
		class_feats = torch.cat(class_feats, dim=0).view((len(classes), 512))
		torch.save(class_feats, save_path)
		return save_path

def extended_avg_token_prompt_features(classes, attributes):
	save_path = "../../prompting/data/tinyimagenet/extended_avg_token_class_prompt_features.pt"
	if os.path.exists(save_path):
		return save_path

	else:
		model, _ = clip.load("ViT-B/16", "cuda")
		template = []
		for c in classes:
			prompt = f"a photo of {c} which has "
			for attr in attributes[c]:
				prompt += attr + ", "
			prompt = prompt[:-1] + "."
			template.append(prompt)
		
		texts = clip.tokenize(template, truncate=True).cuda()
	
		class_feats = []
		with torch.no_grad():
			for i in tqdm(range(texts.shape[0])):
				text_features = encode_text_with_token_avg(model, texts[i].unsqueeze(0)).detach().cpu()
				class_feats.append(text_features.view((1, 512)))
		
		class_feats = torch.cat(class_feats, dim=0).view((len(classes), 512))
		torch.save(class_feats, save_path)
		return save_path

def get_class_prompt_features(classes, mode="basic"):
	if mode == "basic":
		return basic_prompt_features(classes)
	
	elif mode == "extended":
		attributes = torch.load("../../prompting/data/imagenet/imagenet_class_attributes.pt")
		return extended_prompt_features(classes, attributes)
	
	elif mode == "avg_token":
		return avg_token_prompt_features(classes)

	elif mode == "extended+avg_token":
		attributes = torch.load("../../prompting/data/imagenet/imagenet_class_attributes.pt")
		return extended_avg_token_prompt_features(classes, attributes)

def get_descriptor_support(classes, descriptor_database):
	support = []
	for c in classes:
		features = descriptor_database[c]
		support.append(features)

	return support

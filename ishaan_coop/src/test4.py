import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
import argparse
import warnings
from PIL import Image
import os

from data import load_dataset
from utils import initialize_support
from utils import prediction_methods as clip_methods
from utils import training_methods

from utils.augment_clip import augment_clip
from utils.coop_utils import get_class_template_coop_extended

warnings.simplefilter("ignore")


def run(args):
	model, preprocess = clip.load(args.clip_model_name, args.device, jit=False)
	augment_clip(args, model)

	# print(dir(model))
	image_features = model.encode_image(preprocess(Image.open(r'sample_img.png')).to(args.device).unsqueeze(0).repeat(3, 1, 1, 1)) # batch size 3
	image_features = image_features / image_features.norm(dim=-1, keepdim=True)
	
	text_features = model.encode_text_cocoop(clip.tokenize(['sample class text' for _ in range(200)]).to(args.device), image_features) # just checking if this works or not

	# # load a fine-tuned checkpoint (not if you want to run normal CLIP)
	# ckpt = torch.load("../../coco_finetuning/jaisidh_ckpts/model_ft_5_coco.pt")
	# model.load_state_dict(ckpt["model"])
	# model.to(args.device)

	dataset = load_dataset(args.dataset_name, args.dataset_split, args.prompt_method, preprocess) # the dataset the approach will be evaluated on
	train_dataset = load_dataset(args.dataset_name, 'train', args.prompt_method, preprocess)
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
	
	class_templates = get_class_template_coop_extended(args, loader.dataset.classes)
	with open('class_templates.txt', 'w') as f:
		f.writelines([line + '\n' for line in class_templates])

	


if __name__ == "__main__":
	# command line args for ease
	def get_args():
		parser = argparse.ArgumentParser()
		parser.add_argument("--dataset-name", type=str, default="tinyimagenet")
		parser.add_argument("--method", type=str, default="cocoop", choices=["cocoop", "coop", "base", "support", "project_prompts", "extended_prompts"])
		parser.add_argument("--prompt-method", type=str, default="basic", choices=["basic", "extended", "avg_token", "extended+avg_token"])
		parser.add_argument("--device", type=str, default="cuda")
		parser.add_argument("--clip-model-name", type=str, default="ViT-B/16")
		parser.add_argument("--batch-size", type=int, default=2)
		parser.add_argument("--dataset-split", type=str, default="val")
		parser.add_argument("--descriptor-database-path", type=str, default="../../prompting/data/imagenet/imagenet_attribute_features.pt")
		parser.add_argument("--n-ctx", type=int, default=16)
		parser.add_argument("--n-attrib", type=int, default=3) # only implemented in coop extended so far.
		parser.add_argument("--epochs", type=int, default=10)
		parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'])
		parser.add_argument("--lr", type=float, default=0.002) # 2e-5 tha with adam
		parser.add_argument("--checkpoint-name", type=str, default='')
		# parser.add_argument("--n-shots", type=int, default=100) # TO ADD: modify the training function for n-shot training.
		args = parser.parse_args()
		return args

	args = get_args()
	run(args)

    
    
    

from .output_methods import *
import torch
from tqdm import tqdm


def attn_project_prompts(args, prompt_features, support):
	projected_prompts = torch.zeros_like(prompt_features)

	for i in range(prompt_features.shape[0]):
		q = prompt_features[i].unsqueeze(0)
		q /= q.norm(dim=-1, keepdim=True)
		q = q.to(args.device)
		k = support[i]
		k /= k.norm(dim=-1, keepdim=True)
		k = k.to(args.device)
		v = k

		attn = (5 * q @ k.T).softmax(dim=-1)
		projected_prompt = attn @ v
		projected_prompts[i] = projected_prompt

	projected_prompts /= projected_prompts.norm(dim=-1, keepdim=True)
	projected_prompts = projected_prompts.to(args.device)

	return projected_prompts


# default clip classification
def base_clip_classify(args, model, loader, prompt_features):
	correct, total = 0, 0
	bar = tqdm(total=len(loader))
	for (images, labels) in loader:
		images = images.float().to(args.device)
		labels = labels.long().to(args.device)

		with torch.no_grad():
			image_features = model.encode_image(images)
			image_features /= image_features.norm(dim=-1, keepdim=True)

			# get semantic similarity
			similarity = (100 * image_features @ prompt_features.T).softmax(dim=-1)
			predictions = torch.argmax(similarity, dim=1)
			correct += (predictions == labels).sum().item()
			total += labels.shape[0]
	
		accuracy = round(correct/total, 4)
		bar.update(1)
		bar.set_postfix({"accuracy": accuracy})

	bar.close()
	return accuracy

# clip classification using projection with support
def support_clip_classify(args, model, loader, prompt_features, support_features):
	correct, total = 0, 0
	bar = tqdm(total=len(loader))
	for (images, labels) in loader:
		images = images.float().to(args.device)
		labels = labels.long().to(args.device)

		with torch.no_grad():
			image_features = model.encode_image(images)
			image_features /= image_features.norm(dim=-1, keepdim=True)

			# project using support
			sim = (100 * image_features @ support_features.T).softmax(dim=-1)
			projected_features = sim @ support_features
			projected_features /= projected_features.norm(dim=-1, keepdim=True)

			# get semantic similarity
			similarity = (100 * projected_features @ prompt_features.T).softmax(dim=-1)
			predictions = torch.argmax(similarity, dim=1)
			correct += (predictions == labels).sum().item()
			total += labels.shape[0]
	
		accuracy = round(correct/total, 4)
		bar.update(1)
		bar.set_postfix({"accuracy": accuracy})

	bar.close()
	return accuracy

def avg_img_token_clip_classify(args, model, loader, prompt_features):
	correct, total = 0, 0
	bar = tqdm(total=len(loader))
	for (images, labels) in loader:
		images = images.float().to(args.device)
		labels = labels.long().to(args.device)

		with torch.no_grad():
			# image_features = model.encode_image(images)
			image_features = encode_image_with_token_avg(model.visual, images.type(model.dtype))
			image_features /= image_features.norm(dim=-1, keepdim=True)

			# get semantic similarity
			similarity = (100 * image_features @ prompt_features.T).softmax(dim=-1)
			predictions = torch.argmax(similarity, dim=1)
			correct += (predictions == labels).sum().item()
			total += labels.shape[0]
	
		accuracy = round(correct/total, 4)
		bar.update(1)
		bar.set_postfix({"accuracy": accuracy})

	bar.close()
	return accuracy

# base coop classification without support features
# def coop_classify(args, model, loader, prompt_features):

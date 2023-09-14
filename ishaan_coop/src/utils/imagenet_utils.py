from PIL import Image
import os
import torch
from tqdm import tqdm
import clip


device = "cuda"
model, preprocess = clip.load("ViT-B/16", device=device)

def make_imagenet_attributes_database():
	llm_output_folder = "../../data/imagenet/imagenet_class_attributes"
	llm_files = os.listdir(llm_output_folder)

	data = {}
	for filename in tqdm(llm_files):
		class_name = filename.split(".")[0].lower()
		if "," in class_name:
			class_name = " or ".join(class_name.split(",")[:3])

		attributes = []
		path = os.path.join(llm_output_folder, filename)	
		with open(path, encoding="latin-1") as f:
			for line in f.readlines():
				attribute_text = line[2:]
				attributes.append(attribute_text)
		
		data[class_name] = attributes

	torch.save(data, "../../data/imagenet/imagenet_class_attributes.pt")

def encode_text_features():
	data_dir = "../../data/imagenet"
	texts_database_path = os.path.join(data_dir, "imagenet_class_attributes.pt")
	texts_database = torch.load(texts_database_path)

	data = {}
	with torch.no_grad():
		for k, v in tqdm(texts_database.items()):
			texts = clip.tokenize(v).to(device)
			text_features = model.encode_text(texts).cpu()
			data[k] = text_features

	torch.save(data, os.path.join(data_dir, "imagenet_attribute_features.pt"))	

def clip_similarity(image_features, text_features):
	image_features /= image_features.norm(dim=-1, keepdim=True)
	text_features /= text_features.norm(dim=-1, keepdim=True)
	similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
	return similarity[0]

def test_image_with_class():
	# get image features
	test_image_path = "../data/tests/acoustic_guitar.jpg"
	test_image = Image.open(test_image_path).convert("RGB")
	test_image_tensor = preprocess(test_image).unsqueeze(0).to(device)
	test_image_features = model.encode_image(test_image_tensor)

	# features of attributes of the class
	imagenet_attribute_features = torch.load("../data/imagenet/imagenet_attribute_features.pt")
	test_attribute_features = imagenet_attribute_features["acoustic guitar"].to(device)

	# features of the template for classification
	test_class_prompt = "a photo of an acoustic guitar"
	test_class_prompt = clip.tokenize([test_class_prompt]).to(device)
	test_class_features = model.encode_text(test_class_prompt)

	# features projected to text region
	projected_features = test_image_features @ test_attribute_features.T
	projected_features = projected_features @ test_attribute_features

	projected_sim = clip_similarity(projected_features, test_class_features)
	base_sim = clip_similarity(test_image_features, test_class_features)

	ps, _ = projected_sim.topk(1)
	bs, _ = base_sim.topk(1)

	print(f"projected: {ps}")
	print(f"base: {bs}")


if __name__ == "__main__":
	make_imagenet_attributes_database()
	encode_text_features()


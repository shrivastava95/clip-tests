import torch
import clip
from tqdm import tqdm


device = "cuda"
model, preprocess = clip.load("ViT-B/16", device)

def query_prompt_test():
	database = torch.load("../data/imagenet/imagenet_class_attributes.pt")
	classes = database.keys()

	# make class prompts
	template = [f"{c}" for c in classes]
	classes_texts = clip.tokenize(template).to(device)
	class_feats = []
	with torch.no_grad():
		for ct in tqdm(classes_texts):
			feats = model.encode_text(ct.unsqueeze(0))
			class_feats.append(feats.view(1, 512))

	class_feats = torch.cat(class_feats, dim=0)
	class_feats /= class_feats.norm(dim=-1, keepdim=True)

	# make queries following a question-based template
	queries = []
	for c in classes:
		prompt = f"something which has "
		for attr in database[c]:
			prompt += attr + ", "
		prompt = prompt[:-1] + "."
		queries.append(prompt)
	
	query_texts = clip.tokenize(queries, truncate=True).to(device)
	query_feats = []
	with torch.no_grad():
		for ct in tqdm(query_texts):
			feats = model.encode_text(ct.unsqueeze(0))
			query_feats.append(feats.view(1, 512))

	query_feats = torch.cat(query_feats, dim=0)
	query_feats /= query_feats.norm(dim=-1, keepdim=True)

	# start evaluation
	N = query_feats.shape[0]
	labels = torch.tensor([i for i in range(N)]).view(N).to(device)

	sim = (100 * query_feats @ class_feats.T).softmax(dim=-1)
	sim = sim.view(N, N)

	preds = torch.argmax(sim, dim=1)
	correct = (preds == labels).sum().item()
	
	accuracy = round(correct/N, 4)
	print(accuracy)

query_prompt_test()



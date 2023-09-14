import torch
from collections import OrderedDict

def add_trainable_param_coop(args, model):
	# coop: trainable parameter
	ctx_vecs = torch.empty(args.n_ctx, model.transformer.width)
	torch.nn.init.normal_(ctx_vecs, std=0.02) # from the paper
	trainable_param = torch.nn.Parameter(ctx_vecs)
	model.register_parameter('trainable_param', trainable_param)


def add_meta_net_cocoop(args, model):
	# cocoop: meta net
	ctx_dim = model.ln_final.weight.shape[0]
	vis_dim = model.visual.output_dim
	model.meta_net = torch.nn.Sequential(OrderedDict([
		("linear1", torch.nn.Linear(vis_dim, vis_dim // 16)),
		("relu", torch.nn.ReLU(inplace=True)),
		("linear2", torch.nn.Linear(vis_dim // 16, ctx_dim))
	]))
	model.meta_net = model.meta_net.half() # is this okay? check if this is causing problems


def define_coop_encoder(args, model):
	def encode_text_coop(text):
		x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model]

		# ishaan: line added here for the replacement using the extra parameter stored inside the model
		context_embedding = model.trainable_param.unsqueeze(0).repeat(x.shape[0], 1, 1)
		x[:, 1:1+context_embedding.shape[1], :] = context_embedding # tokenizer outputs: <start> <word1> <word2> <word3> ... <eos> <0> <0> ... 77 tokens in total

		x = x + model.positional_embedding.type(model.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = model.transformer(x)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = model.ln_final(x).type(model.dtype)

		# x.shape = [batch_size, n_ctx, transformer.width]
		# take features from the eot embedding (eot_token is the highest number in each sequence)
		x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection

		return x
	model.encode_text_coop = encode_text_coop

def define_cocoop_encoder(args, model):
	def encode_text_cocoop(text, image_features_normed):
		bx = model.token_embedding(text).type(model.dtype)  # [num_classes, n_tokens, d_model]
		bx = bx.unsqueeze(0).repeat(image_features_normed.shape[0], 1, 1, 1) # [batch_size, num_classes, n_tokens, d_model] or BNLD

		# ishaan: line added here for the replacement using the extra parameter stored inside the model
		context_embedding = model.trainable_param              # [n_ctx, d_model]
		meta_embedding = model.meta_net(image_features_normed) # [batch_size, d_model]
		context_embedding = context_embedding.unsqueeze(0)     # [1, n_ctx, d_model]
		meta_embedding = meta_embedding.unsqueeze(1)           # [batch_size, 1, d_model]
		net_embedding = context_embedding + meta_embedding     # [batch_size, n_ctx, d_model]
		net_embedding = net_embedding.unsqueeze(1).repeat(1, bx.shape[1], 1, 1) # [batch_size, num_classes, n_ctx, d_model] or BNnD
		
		bx[:, :, 1:1+context_embedding.shape[1], :] = net_embedding # tokenizer outputs: <start> <word1> <word2> <word3> ... <eos> <0> <0> ... 77 tokens in total

		text_features = []
		for x in bx:
			x = x + model.positional_embedding.type(model.dtype)
			x = x.permute(1, 0, 2)  # NLD -> LND
			x = model.transformer(x)
			x = x.permute(1, 0, 2)  # LND -> NLD
			x = model.ln_final(x).type(model.dtype)

			# x.shape = [num_classes, n_ctx, transformer.width]
			# take features from the eot embedding (eot_token is the highest number in each sequence)
			x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection

			text_features.append(x)
		text_features = torch.stack(text_features)

		return text_features
	model.encode_text_cocoop = encode_text_cocoop


def augment_clip(args, model):

	# augment CLIP model
	add_trainable_param_coop(args, model)
	add_meta_net_cocoop(args, model)

	# define the modified text encode methods
	define_coop_encoder(args, model)
	define_cocoop_encoder(args, model)
	model = model.to(args.device)
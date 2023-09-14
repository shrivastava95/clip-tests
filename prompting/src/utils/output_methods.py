# tinkering with CLIP's architecture
import torch


def encode_text_with_token_avg(model, text):
	batch_size = text.shape[0]

	x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model]

	x = x + model.positional_embedding.type(model.dtype)
	x = x.permute(1, 0, 2)  # NLD -> LND
	x = model.transformer(x)
	x = x.permute(1, 0, 2)  # LND -> NLD
	x = model.ln_final(x).type(model.dtype)

	transformer_dim = x.shape[-1]
	# x.shape = [batch_size, n_ctx, transformer.width]

	# DEFAULT BEHAVIOUR:
	# take features from the eot embedding (eot_token is the highest number in each sequence)
	# x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection

	# MODIFIED BEHAVIOUR:
	# take features from all the tokens via mean
	start_index = 38
	y = x.mean(dim=1).view(batch_size, transformer_dim)
	y = y @ model.text_projection

	return y

def encode_image_with_token_avg(self, x):
	x = self.conv1(x)  # shape = [*, width, grid, grid]
	x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
	x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
	x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
	x = x + self.positional_embedding.to(x.dtype)
	x = self.ln_pre(x)

	x = x.permute(1, 0, 2)  # NLD -> LND
	x = self.transformer(x)
	x = x.permute(1, 0, 2)  # LND -> NLD

	batch_size, transformer_dim = x.shape[0], x.shape[-1]

	# x = self.ln_post(x[:, 0, :])
	x = x.mean(dim=1).view(batch_size, transformer_dim)

	if self.proj is not None:
		x = x @ self.proj

	return x
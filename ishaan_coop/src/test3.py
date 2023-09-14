import torch
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

# n_ctx = 16
model, preprocess = clip.load('ViT-B/16', device='cuda', jit=False)
# trainable_param = torch.nn.Parameter(torch.randn(n_ctx, model.transformer.width))
# model.register_parameter('trainable_param', trainable_param)
trainable_param = torch.nn.Parameter()

sent1 = "sample cat"
sent2 = "sample dog" 

print(clip.tokenize([sent1, sent2]))
# print()
# emb1 = (model.token_embedding(clip.tokenize([sent1]).to(device)))
# emb2 = (model.token_embedding(clip.tokenize([sent2]).to(device)))
# diff = (emb1 - emb2)
# print(diff[:, 5])

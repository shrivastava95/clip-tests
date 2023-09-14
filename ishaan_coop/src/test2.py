import torch

path = '../data/tinyimagenet/train_dataset_helper.pt'
stuff = torch.load(path)

print(list(stuff.keys())[0])

new_stuff = {key.replace('../data', '../data'): value for key, value in stuff.items()}
# # torch.save(new_stuff, path.replace('../../prompting/data', '../data'))
torch.save(new_stuff, path)
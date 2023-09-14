# ishaan: enter the training methods for clip that you will call from main.py here

import torch
from tqdm import tqdm
import clip
from torch.optim import Adam, SGD
from .coop_utils import get_class_template_coop_extended, get_class_template_coop


def build_optim_scheduler(args, model):
    params = [model.trainable_param] + list(model.meta_net.parameters())

    if args.optim == 'adam':
        optimizer = Adam(params, lr=args.lr)

    elif args.optim == 'sgd':
        optimizer = SGD(params, lr=args.lr)

    else:
        assert False, f"{args.optim}: Not a valid optimizer"
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer, float(args.epochs) # max_epoch = 1 in our case as we are warming up only one epoch.
	)

    return optimizer, scheduler
    # scheduler not yet implemented. 
    # based on the official implementation 
    # i believe they never used a scheduler 
    # in the forward pass despite saying that they did.
    

def coop_clip_train(args, model, loader, prompt_features, optimizer, scheduler, criterion):
    correct, total = 0, 0
    bar = tqdm(total=len(loader))
    if args.prompt_method == 'basic':
        class_texts_template = get_class_template_coop(args, loader.dataset.classes)
    elif args.prompt_method == 'extended':
        class_texts_template = get_class_template_coop_extended(args, loader.dataset.classes)
    for (images, labels) in loader:
        images = images.float().to(args.device)
        labels = labels.long().to(args.device)

        # get image features
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # get text features
        class_features = model.encode_text_coop(clip.tokenize(class_texts_template).to(args.device)) # using the modded text encoder: coop
        similarity = (100 * image_features @ class_features.T).softmax(dim=-1)
        predictions = torch.argmax(similarity, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.shape[0]
        
        # backprop the loss
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(similarity, labels)
        loss.backward()
        optimizer.step()

        accuracy = round(correct/total, 4)
        bar.update(1)
        bar.set_postfix({"accuracy": accuracy})
    scheduler.step()
    
    bar.close()

def cocoop_clip_train(args, model, loader, prompt_features, optimizer, scheduler, criterion):
    correct, total = 0, 0
    bar = tqdm(total=len(loader))
    class_texts_template = ['sample ' * args.n_ctx + loader.dataset.classes[class_idx] for class_idx in range(len(loader.dataset.classes))]
    for images, labels in loader:
        images = images.float().to(args.device)
        labels = labels.long().to(args.device)

        # get image features
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # get text features
        class_features = model.encode_text_cocoop(clip.tokenize(class_texts_template).to(args.device), image_features)  # using the modded text encoder: cocoop
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        
        # get similarity
        image_features = image_features.unsqueeze(2)
        similarity = torch.bmm(class_features, 100 * image_features).squeeze(2).softmax(dim=-1)
        predictions = torch.argmax(similarity, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.shape[0]

        # backprop the loss
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(similarity, labels)
        loss.backward()
        optimizer.step()

        accuracy = round(correct/total, 4)
        bar.update(1)
        bar.set_postfix({"accuracy": accuracy})
    scheduler.step()
    
    bar.close()
import os
import clip, torch

def get_class_attributes_text(classes):
	save_path = "../../prompting/data/imagenet/imagenet_class_attributes.pt"
	if os.path.exists(save_path):
		imagenet_class_attributes = torch.load(save_path)
		attributes = []
		for c in classes:
			attributes.append(imagenet_class_attributes[c])
		return attributes
	else:
		assert False

def get_class_template_coop(args, classes):
	return ['sample ' * args.n_ctx + classes[class_idx] for class_idx in range(len(classes))]

def get_class_template_coop_extended(args, classes):
	attributes = get_class_attributes_text(classes)
	class_texts_template = []
	for class_idx in range(len(classes)):
		attrib_count = args.n_attrib
		while True:
			template = 'sample ' * args.n_ctx + \
                        classes[class_idx] + \
						', attributes: ' + \
						', '.join(([item.strip(' \n') for item in attributes[class_idx][:attrib_count]]))
			try:
				clip.tokenize(template)
				class_texts_template.append(template)
				break
			except:
				attrib_count -= 1
				if attrib_count == 0:
					assert False, "something wrong with the attributes."
					
				
	# class_texts_template = ['sample ' * args.n_ctx 
	# 					   + classes[class_idx] 
	# 					   + ', attributes: '
	# 					   + ', '.join(attributes[class_idx][:args.n_attrib]) # take the first n_attrib attributes 
	# 					   for class_idx in range(len(classes))]
	return class_texts_template
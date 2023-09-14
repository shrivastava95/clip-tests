from .tinyimagenet import *

def load_dataset(dataset_name, split, mode, transform):
	mapping = {
    	"tinyimagenet": TinyImageNetDataset
	}
	return mapping[dataset_name](split, mode, transform)
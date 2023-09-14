from .tinyimagenet_utils import get_descriptor_support


def initialize_support(dataset_name, classes, descriptor_database):
    mapping = {
        "tinyimagenet": get_descriptor_support,
	}
    return mapping[dataset_name](classes, descriptor_database)
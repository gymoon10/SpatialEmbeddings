from src.datasets.CityscapesDataset import CityscapesDataset
from src.datasets.CVPPPDataset import CVPPPDataset
from src.datasets.CVPPPDataset2 import CVPPPDataset2

def get_dataset(name, dataset_opts):
    if name == "cityscapes": 
        return CityscapesDataset(**dataset_opts)
    elif name == 'cvppp':
        return CVPPPDataset(**dataset_opts)
    elif name == 'cvppp2':
        return CVPPPDataset2(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))
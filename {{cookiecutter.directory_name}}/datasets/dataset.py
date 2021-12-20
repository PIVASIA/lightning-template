from torch.utils.data import Dataset

# TODO: ADD YOUR CODE HERE
class {{cookiecutter.dataset_name}}(Dataset):
    def __init__(self,
                 transform=None):
        self.transform = transform
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
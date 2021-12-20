from torch.utils.data import Dataset

class {{cookiecutter.dataset_name}}(Dataset):
    def __init__(self,
                 transform=None):
        # TODO: ADD YOUR CODE HERE
        self.transform = transform
    
    def __len__(self):
        # TODO: ADD YOUR CODE HERE
        pass
    
    def __getitem__(self, idx):
        # TODO: ADD YOUR CODE HERE
        pass
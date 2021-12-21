import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split

from datasets.dataset import {{cookiecutter.dataset_name}}


# TODO: ADD YOUR CODE TO READ DATA 
def _read_data(filepath):
    pass


class {{cookiecutter.datamodule_name}}(pl.LightningDataModule):
    def __init__(self,
                 train_path: str,
                 test_path: str,
                 train_batch_size=32,
                 test_batch_size=16,
                 seed=28):
        super().__init__()

        self.train_path = train_path
        self.test_path = test_path

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.seed = seed

    def prepare_data(self):
        pass
    
    def setup(self, stage="fit"):
        # TODO: ADD ANY TRANSFORM YOU NEED
        train_transform = transforms.Compose([])
        test_transform = transforms.Compose([])

        if stage == "fit" or stage is None:
            images, labels = _read_data(self.train_path)
            x_train, x_val, y_train, y_val =\
                train_test_split(images, labels, test_size=0.3, random_state=self.seed)

            self.train_dataset = {{cookiecutter.dataset_name}}(x_train,
                                            y_train, 
                                            transform=train_transform)
            self.val_dataset = {{cookiecutter.dataset_name}}(x_val,
                                          y_val,
                                          transform=test_transform)

        if stage == "predict" or stage is None:
            images, labels = _read_data(self.test_path)
            self.test_dataset = {{cookiecutter.dataset_name}}(images,
                                           labels, 
                                           transform=test_transform)

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(self.train_dataset,
                              batch_size=self.train_batch_size,
                              shuffle=True, 
                              num_workers=2)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_worker=2)

    def predict_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_worker=2)
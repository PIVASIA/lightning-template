import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class {{cookiecutter.model_name}}(pl.LightningModule):
    # TODO: ADD YOUR PARAMETERS AS MANY AS YOU NEED
    def __init__(self,
                 n_epochs: int = 30, 
                 lr: float = 1e-5,
                 weight_decay: float = 0.01,
                 momentum: float = 0.9):
        super().__init__()
        self.save_hyperparameters()
        
        # TODO: ADD YOUR MODEL INITIALIZATION
        self.model = None
        
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        # TODO: ADD YOUR LOSS FUNCTION
        self.criterion = None

    def forward(self, x):
        output = self.model(x)
        return output
    
    def configure_optimizers(self):
        # TODO: REPLACE WITH YOUR DESIRED OPTIMIZER
        optimizer = optim.SGD(self.parameters(), 
                              lr=self.lr, 
                              momentum=self.momentum)
        return optimizer 

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        self.log('val_loss', loss, prog_bar=True, logger=True)


class {{cookiecutter.model_task}}(pl.LightningModule):
    def __init__(self,
                 model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output

    def postprocess(self, logits):
        """
        Post-process for model output.
        E.g. For classification task, apply `torch.sigmoid()` or `torch.softmax()` 
        function to model output in case of training with `nn.BCEWithLogitsLoss()`.

        Args:
            logits ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return logits

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        # TODO: ADD YOUR INFERENCE
        logits = self(batch['inputs'])
        logits = self.postprocess(logits)

        if "label" in batch.keys():
            return logits.data.cpu().numpy(), batch["label"].data.cpu().numpy()

        return logits.data.cpu().numpy()
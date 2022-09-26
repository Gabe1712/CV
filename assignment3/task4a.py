import pathlib
import matplotlib.pyplot as plt
import utils
from torch import max_pool2d, nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from task2 import ExampleModel, create_plots
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import PIL

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import torchvision

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def get_data_dir():
    server_dir = pathlib.Path("/work/datasets/cifar10")
    if server_dir.is_dir():
        return str(server_dir)
    return "data/cifar10"


def load_cifar10(batch_size: int, validation_fraction: float = 0.1
                 ) -> typing.List[torch.utils.data.DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    data_train = datasets.CIFAR10(get_data_dir(),
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    data_test = datasets.CIFAR10(get_data_dir(),
                                 train=False,
                                 download=True,
                                 transform=transform_test)

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True) 
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
                                             # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters 
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected 
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional 
            param.requires_grad = True # layers
    def forward(self, x):
        x = self.model(x)
        return x
    
    
def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 5
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size) #edited
    model = Model()
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task4a")
    
    #verify best view
    trainer.load_best_model()
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    train_loss, train_accuracy = compute_loss_and_accuracy(dataloader_train, trainer.model, nn.CrossEntropyLoss())
    val_loss, val_accuracy = compute_loss_and_accuracy(dataloader_val, trainer.model, nn.CrossEntropyLoss())
    test_loss, test_accuracy = compute_loss_and_accuracy(dataloader_test, trainer.model, nn.CrossEntropyLoss())
    
    print("\nBest Model: ")
    print("Train_Acc=\t",train_accuracy )
    print("Val_Acc=\t",val_accuracy )
    print("Test_Acc=\t",test_accuracy )
    
    
    
if __name__ == "__main__":
    main()
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
mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)

def load_cifar10_augmented(batch_size: int, validation_fraction: float = 0.1
                 ) -> typing.List[torch.utils.data.DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train)
    
    #augmented_data_train = datasets.CIFAR10('data/cifar10',
    #                              train=True,
    #                              download=True,
    #                              transform=transform_train_Augmented)

    data_test = datasets.CIFAR10('data/cifar10',
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


class Model1(nn.Module):
    def __init__(self,
                 image_channels,
                 num_classes,):
        
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        print(num_classes)
        batch_normalization = True
        drop_out =True
        dropout =0.05
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            #Layer 1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            #nn.Softmax(dim=2),
            nn.BatchNorm2d(num_features=num_filters) if batch_normalization else nn.Identity(),
            #nn.MaxPool2d([2, 2], stride=2),
            
            #Layer 2
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            #nn.Softmax(dim=2),
            nn.BatchNorm2d(num_features=num_filters*2) if batch_normalization else nn.Identity(),
            nn.MaxPool2d([2, 2], stride=2),
            #nn.Dropout2d(p=dropout) if drop_out else nn.Identity(),
            

            #Layer 4
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            #nn.Softmax(dim=2),
            nn.MaxPool2d([2, 2], stride=2),
            #nn.BatchNorm2d(num_features=num_filters*4) if batch_normalization else nn.Identity(),
            nn.Dropout2d(p=dropout) if drop_out else nn.Identity(),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 128*8*8 #991232 with k=3 #128*8*8
        self.classifier = nn.Sequential(
        nn.Linear(self.num_output_features, 64),
        nn.BatchNorm1d(64) if batch_normalization else nn.Identity(),
        nn.ReLU(),
        #nn.Softmax(dim=1),

        nn.Linear(64, num_classes),
        nn.Flatten()
        )
        
        
    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        #out = x
        features = self.feature_extractor(x)
        linear_features = features.view(-1, self.num_output_features)
        out = self.classifier(linear_features)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
    def initialize_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('relu'))
            m.bias.data.fill_(0.01)
    

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64#64
    learning_rate = 0.1 #4e-2
    early_stop_count = 4
    dataloaders = load_cifar10_augmented(batch_size) #edited
    model = Model1(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task3_Improved")
    
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
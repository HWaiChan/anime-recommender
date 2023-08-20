import numpy
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Lambda


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ReviewDataSet(Dataset):

    # This will contain the dataset of
    # userid, animeid and the label

    def __init__(self, csv_file, transform=None, target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        print(sample)
        label = None
        # review_data = self.data[]
        # You can implement data preprocessing or transformations here
        if self.transform:
            # Might need to do a one hot encode of the animeid and userid
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(sample)

        return sample, label


if __name__ == '__main__':

    # review_dir = "data/myanimelist/old_data/ratings.csv"
    # data = ReviewDataSet(csv_file=review_dir, )
    # print(data[1])
    # Create data loaders.
    # train_dataloader = DataLoader(training_data, batch_size=batch_size)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size)
    #
    # model = NeuralNetwork().to(device)
    # print(model)
    # X, y = test_data[1][0], test_data[1][1]
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    #
    # print(f'Predicted class: {classes[y_pred]}. Actual: "{classes[y]}"')
    #
    # X_numpy = X.numpy()
    # twoD_X = numpy.reshape(X_numpy, (28,28))
    # plt.imshow(twoD_X, cmap='hot', interpolation='nearest')
    # plt.show()
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # epochs = 5
    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer)
    #     test(test_dataloader, model, loss_fn)
    # print("Done!")
    # torch.save(model.state_dict(), "model.pth")
    # print("Saved PyTorch Model State to model.pth")

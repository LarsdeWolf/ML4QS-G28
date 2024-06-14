import load_data
import torch
import torch.nn as nn
import numpy as np
from utils import np_from_df, train_test_split_measurementlevel
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset


class MLDataset(Dataset):
    def __init__(self, x, y, return_tensor=True):
        self.X = x
        self.y = y
        self.tensor = return_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.X[idx]
        labels = self.y[idx]
        if self.tensor:
            features, labels = torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
        return features, labels


class LSTM(nn.Module):
    """
    LSTM model with linear layer for classification
    """
    def __init__(self, input_size,  hidden_size, layers, out_size, dropout=0.0):
        super(LSTM, self).__init__()
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.lstm = nn.LSTM(input_size, hidden_size, layers, dropout=dropout, batch_first=True, device=self.device)
        self.linear = nn.Linear(hidden_size, out_size, device=self.device)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)  # Pass through LSTM layer
        return self.linear(hidden[-1])   # Pass through linear layer


def accuracy(pred: torch.tensor, gt: torch.tensor):
    preds = pred.argmax(1)
    corr = (preds == gt).sum()
    return corr/len(preds)


def eval(m, dataloader):
    correct = 0
    tot = 0
    m.eval()
    with torch.no_grad():
        for x, y in dataloader:
            out = m(x.to(m.device))
            preds = torch.argmax(out, 1)
            labels = y.squeeze().to(torch.int64).to(m.device)
            correct += (preds == labels).sum()
            tot += len(y)
    return correct / tot


def train(data, step_size=100, epochs=20, lr=1e-3, hidden_size=200, layers=3, labels=5, dropout=0.5):
    """
    Trains a LSTM model the data.
    Uses a CrossEntropyLoss - Negative-Log-Likelihood between log softmax input probs and target labels
    Evaluates each epoch on DEV set and saves the best model
    Args:
        data: Output of load_data.process_data() (normal data or resampled data)
        step_size: Amount of data points in a sequence
        epochs: Number of epochs to train
        lr: Learning rate
        hidden_size: Amount of hidden units per layer
        layers: Amount of hidden (LSTM) layers
        labels: Amount of classed to predict
        dropout: Dropout percentage

    Returns:
        Best model evaluated on the DEV set

    """
    X, y = np_from_df(list(data[i].dropna() for i in range(len(data))
                           if len(data[i].columns) == 23), step_size)
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_split_measurementlevel(X, y)
    train_dataloader = DataLoader(MLDataset(X_train, y_train), batch_size=312, shuffle=True)
    test_dataloader = DataLoader(MLDataset(X_test, y_test), batch_size=64, shuffle=True)
    dev_dataloader = DataLoader(MLDataset(X_val, y_val), batch_size=64, shuffle=True)
    del X, y, X_train, y_train, X_test, y_test, X_val, y_val

    model = LSTM(16, hidden_size, layers, labels, dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    bestdev, best_model, name, steps = 0, None, "", (len(train_dataloader) // 5)
    for EPOCH in range(epochs):
        print("#######################TRAIN#######################")
        model.train()
        loss_avg, accuracy_avg = 0, 0
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            out = model(x.to(model.device))
            labels = y.squeeze().to(torch.int64).to(model.device)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            accuracy_avg += accuracy(out, labels)
            loss_avg += loss.item()
            if (i + 1) % steps == 0:
                loss_avg = loss_avg / steps  # loss per batch
                accuracy_avg = accuracy_avg / steps
                print(f"| EPOCH {EPOCH + 1} | progress: {int(100 * ((i + 1)/len(train_dataloader)))}% | avg loss: {loss_avg} | avg acc: {accuracy_avg} |")
                loss_avg, accuracy_avg = 0, 0

        print("#######################DEV#######################")
        acc_dev = eval(model, dev_dataloader)
        print(f"DEV ACCURACY: {acc_dev}")
        if acc_dev > bestdev:
            print("New Best DEV")
            name = f"devacc-{round(acc_dev.item(), 4)}_EPOCH-{epochs}_lr-{lr}_stepsize-{step_size}_hidden-{hidden_size}_layers-{layers}"
            bestdev = acc_dev
            best_model = deepcopy(model.state_dict())
        print()

    print("#######################TESTING#######################")
    model.load_state_dict(best_model)
    acc_test = eval(model, test_dataloader)
    print(f"TEST ACCURACY: {acc_test}")
    return model, name


if __name__ == '__main__':
    _, data_resampled = load_data.process_data()
    data_resampled = data_resampled['100ms']
    model, name = train(data_resampled)
    torch.save(model.state_dict(), f'models/{name}')








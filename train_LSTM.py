import load_data
import torch
import torch.nn as nn
import numpy as np
from utils import get_data
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast


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


def train(data, epochs=20, lr=1e-3, hidden_size=200, layers=3, labels=5, dropout=0.5):
    X_train, y_train, X_test, y_test, X_val, y_val = data
    train_dataloader = DataLoader(MLDataset(X_train, y_train), pin_memory=True, num_workers=4, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(MLDataset(X_test, y_test), pin_memory=True, num_workers=1, batch_size=64, shuffle=False)
    dev_dataloader = DataLoader(MLDataset(X_val, y_val), pin_memory=True, num_workers=1, batch_size=64, shuffle=False)
    del X_train, y_train, X_test, y_test, X_val, y_val

    model = LSTM(16, hidden_size, layers, labels, dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    bestdev, best_model, name, steps = 0, None, "", (len(train_dataloader) // 5)
    for EPOCH in range(epochs):
        print("#######################TRAIN#######################")
        model.train()
        loss_avg, accuracy_avg = 0, 0
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(model.device), y.to(model.device)
            labels = y.squeeze().to(torch.int64)
            optimizer.zero_grad()
            with autocast():
                out = model(x)
                loss = loss_fn(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            name = f"devacc-{round(acc_dev.item(), 4)}_EPOCH-{epochs}_lr-{lr}_hidden-{hidden_size}_layers-{layers}"
            bestdev = acc_dev
            best_model = deepcopy(model.state_dict())

    print("#######################TESTING#######################")
    model.load_state_dict(best_model)
    acc_test = eval(model, test_dataloader)
    print(f"TEST ACCURACY: {acc_test}")
    return model, name


if __name__ == '__main__':
    dataset_level = 'measurement'  # Or activity
    step_size = 10
    _, data_resampled = load_data.process_data()
    data = data_resampled['100ms']
    sensors = ['Accelerometer', 'Lin-Acc', 'Gyroscope', 'Location']
    data = get_data(data, sensors, dataset_level, 'LSTM', step_size, True, True)
    model, name = train(data, epochs=10)
    torch.save(model.state_dict(), f'models/{name}')







import pickle
import os
import torch
import torch.nn as nn
class LSTM(nn.Module):
    """
    LSTM model with linear layer for classification
    """
    def __init__(self, input_size,  hidden_size, layers, out_size, dropout=0.0):
        super(LSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.lstm = nn.LSTM(input_size, hidden_size, layers, dropout=dropout, batch_first=True, device=self.device)
        self.linear = nn.Linear(hidden_size, out_size, device=self.device)
        self.to(self.device)
    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)  # Pass through LSTM layer
        return self.linear(hidden[-1])   # Pass through linear layer

def load_metrics(path):
    with open(path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def load_model(path, model_type=None):
    if model_type == 'LSTM':
        hidden_size = 200
        layers = 3
        labels = 5
        dropout = 0.5
        model = LSTM(16, hidden_size, layers, labels, dropout)
        model.load_state_dict(torch.load(path))
    else:
        with open(path, 'rb') as f:
            model = pickle.load(f)
    return model
def view_metrics(metrics):
    accuracy, precision, recall, f1 = metrics
    print("================== METRICS ==================")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def view_model(model):
    print("================== MODEL ==================")
    if isinstance(model, torch.nn.Module):
        print(model)
    else:
        if hasattr(model, 'get_params'):
            print("Model Parameters:")
            print(model.get_params())


def view_results(results_dir):
    for root, dirs, files in os.walk(results_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            print("================== LOAD ==================")
            print(f"Processing directory: {dir_path}")
            metrics_path = os.path.join(dir_path, 'metrics.pkl')
            # model_path = os.path.join(dir_path, 'model.pth') if 'LSTM' in dir else os.path.join(dir_path, 'model.pkl')
            model_path = os.path.join(dir_path, 'model.pkl')

            if os.path.exists(model_path):
                if 'LSTM' in dir:
                    # print(f"Loading LSTM model from {model_path}")
                    model = load_model(model_path, 'LSTM')
                elif 'DT' in dir:
                    # print(f"Loading Decision Tree model from {model_path}")
                    model = load_model(model_path)
                elif 'KNN' in dir:
                    # print(f"Loading KNN model from {model_path}")
                    model = load_model(model_path)
                view_model(model)
            if os.path.exists(metrics_path):
                # print(f"Loading metrics from {metrics_path}")
                metrics = load_metrics(metrics_path)
                view_metrics(metrics)



if __name__ == '__main__':
    results_dir = 'Results'
    best_dir = 'best'
    #view_results(results_dir)
    view_results(best_dir)


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from table import Table

class Learner(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        current_size = input_size
        for h_size in hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(current_size, h_size))
            self.hidden_layers.append(nn.ReLU())
            current_size = h_size
        self.output_layer = nn.Linear(current_size, 1)

    def forward(self, x, return_penultimate=False):
        penultimate = x
        for layer in self.hidden_layers:
            penultimate = layer(penultimate)
        output = self.output_layer(penultimate)
        if return_penultimate:
            return output, penultimate
        else:
            return output

class GrowNet(nn.Module):
    def __init__(self, num_learners=10, hidden_layer_sizes=(16,), learning_rate_init=0.01, table=None):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.num_learners = num_learners
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.table = table

        X_features = self.table.X.shape[1]
        penultimate_size = hidden_layer_sizes[-1]

        self.learners = nn.ModuleList()
        self.alphas = nn.ParameterList()

        for i in range(num_learners):
            input_size = X_features if i == 0 else X_features + penultimate_size
            learner = Learner(input_size, hidden_layer_sizes).to(self.device)
            self.learners.append(learner)
            self.alphas.append(nn.Parameter(torch.tensor(1.0, device=self.device)))

    def forward(self, X):
        X = X.to(self.device)
        total_output = torch.zeros(X.size(0), 1, device=self.device)
        prev_penultimate = None

        for i, (learner, alpha) in enumerate(zip(self.learners, self.alphas)):
            if i == 0:
                inputs = X
            else:
                inputs = torch.cat([X, prev_penultimate], dim=1)
            output, penultimate = learner(inputs, return_penultimate=True)
            total_output += alpha * output
            prev_penultimate = penultimate

        return total_output
    


# Load data using the Table class
table = Table("Table.csv")
table.encode(table.non_integer_columns())  # Encode categorical columns if any

# Convert data to PyTorch tensors
X_tensor = torch.tensor(table.X.values, dtype=torch.float32)
y_tensor = torch.tensor(table.y.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize GrowNet
model = GrowNet(num_learners=10, hidden_layer_sizes=(16, 16), table=table)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.to(model.device))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
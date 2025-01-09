from table import Table
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

class GrowNet:
    def __init__(self, num_learners=10, hidden_layer_sizes=(16,), learning_rate_init=0.01, table = Table("X.csv")):
        self.num_learners = num_learners
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.learners = []  # List to store weak learners
        self.alphas = []  # Weights for learners
        for i in range(num_learners):
            if i == 0:
                # initialise a learner with input layer of size 
                pass
        print(f"Initialized GrowNet with {self.num_learners} learners, hidden layer size {self.hidden_layer_sizes}, and learning rate {self.learning_rate_init}")

    def fit(self, X, y):
        residual = y.copy()
        augmented_X = X.copy()  # Start with original features
        print("Starting training process...")
        for i in range(self.num_learners):
            print(f"Training learner {i+1}/{self.num_learners}...")
            learner = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, 
                                   max_iter=1, learning_rate_init=self.learning_rate_init, 
                                   warm_start=True, solver='adam')

            learner.fit(augmented_X, residual)
            predictions = learner.predict(augmented_X)

            error = mean_squared_error(residual, predictions)
            alpha = 1 if error == 0 else 1 / (1 + error)
            self.alphas.append(alpha)
            print(f"Learner {i+1} - Error: {error:.6f}, Alpha: {alpha:.6f}")

            residual -= alpha * predictions

            penultimate_output = learner.predict(augmented_X)  
            penultimate_output = penultimate_output.reshape(-1, 1) 
            
            augmented_X = np.hstack((augmented_X, penultimate_output))g
            self.learners.append(learner)
        print("Training complete.")

    def predict(self, X):
        print("Making predictions...")
        predictions = np.zeros(X.shape[0])
        augmented_X = X.copy()  # Start with original features
        for i, (alpha, learner) in enumerate(zip(self.alphas, self.learners)):
            learner_predictions = learner.predict(augmented_X)
            predictions += alpha * learner_predictions
            print(f"Learner {i+1} contribution added to predictions.")

            penultimate_output = learner.predict(augmented_X)  
            penultimate_output = penultimate_output.reshape(-1, 1)  
            augmented_X = np.hstack((augmented_X, penultimate_output))
        return predictions

if __name__ == "__main__":
    print("Loading data...")
    table = Table("X.csv") 

    grownet = GrowNet(num_learners=10, hidden_layer_sizes=(16,), learning_rate_init=0.01)

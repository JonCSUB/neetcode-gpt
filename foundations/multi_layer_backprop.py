import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Forward pass
        z1 = np.dot(W1, x) + b1 # first layer linear transformation
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = np.dot(W2, a1) + b2 # second layer linear transformation
        predictions = z2  # y hat

        # Compute loss
        loss = np.mean((predictions - y_true) ** 2)

        # now do backward pass
        dL_dpredictions = 2 * (predictions - y_true) / len(y_true)  # derivative of MSE loss
        dL_dz2 = dL_dpredictions  # since z2 is the output layer, the derivative is the same
        dL_dW2 = np.outer(dL_dz2, a1) 

        dL_db2 = dL_dz2 

        dL_da1 = np.dot(W2.T, dL_dz2)  # backprop through second layer
        dL_dz1 = dL_da1 * (z1 > 0)

        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return {
            'loss': round(loss, 4),
            'dW1': [[round(val, 4) for val in row] for row in dL_dW1],
            'db1': [round(val, 4) for val in dL_db1],
            'dW2': [[round(val, 4) for val in row] for row in dL_dW2],
            'db2': [round(val, 4) for val in dL_db2]
        }
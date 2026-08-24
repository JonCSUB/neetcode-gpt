import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        # Forward pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z)) 
        L = 1/2 * (y_hat - y_true) ** 2 

        # Backward pass
        dL_dy_hat = y_hat - y_true
        dy_hat_dz = y_hat * (1 - y_hat)  # derivative of sigmoid
        dz_dw = x
        dz_db = 1 

        dL_dw = dL_dy_hat * dy_hat_dz * dz_dw
        dL_db = dL_dy_hat * dy_hat_dz * dz_db

        return np.round(dL_dw, 5), round(dL_db, 5)
"""Module containing abstract Optimizer class and optimizer implementations."""
import numpy as np


class Optimizer:
    """Abstract Optimizer class."""

    def __init__(self, learning_rate: float, weight_decay: float) -> None:
        """Initialize optimizer parameters.

        Args:
            learning_rate (float): Optimizer's learning rate.
            weight_decay (float): Optimizer's weight decay.

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, params: list, grads: list) -> list:
        """Update parameters.

        Args:
            params (list): List of parameters to update.
            grads (list): Corresponding gradients.

        Returns:
            list: Updated parameters.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Attributes:
        learning_rate (float, optional): Learning rate for training. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay to hinder overfitting. Defaults to 0.
        momentum (float, optional): Momentum parameter >= 0 which dampens oscillations
            during optimization. Defaults to 0 (vanilla SGD).
    """

    def __init__(self, learning_rate=1e-3, weight_decay=0.0, momentum=0.0):
        """Initialize SGD optimizer. See help(SGD) for more information."""
        self.velocity = None
        self.momentum = momentum
        super().__init__(learning_rate, weight_decay)

    def update(self, params, grads):
        """Update parameters.

        For SGD with momentum the weights are updated according to
        velocity = momentum * velocity - learning_rate * (gradient + weight_decay * parameter)
        parameter = parameter + velocity

        Args:
            params (list): List of np.ndarray containing the parameters to update.
            grads (list): List of np.ndarray with the corresponding gradients.

        Returns:
            params (list): List of np.ndarray containing the updated parameters.
        """
        if self.velocity is None:
            self.velocity = [np.zeros(param.shape) for param in params]

        for idx, (vel, param, grad) in enumerate(zip(self.velocity, params, grads)):
            vel = self.momentum * vel - self.learning_rate * (
                grad + self.weight_decay * param
            )
            param += vel
            self.velocity[idx] = vel

        return params

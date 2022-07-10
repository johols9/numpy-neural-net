"""This module contains the class Dense, which is a fully connected layer."""
import numpy as np

from layers.base_layer import Layer
from optimizers import Optimizer


class Dense(Layer):
    """Dense / Fully connected layer.

    Attributes:
        outputs (int): Positive integer specifying the number of output nodes in the layer.
        weights (np.ndarray, optional): Weights matrix with shape (width*heigth*channels, outputs).
            If None, weights are initalized with He-initialization.
        bias (np.ndarray, optional): Bias vector with shape (outputs,).
            If None, bias is initialized as a zero vector.

    Raises:
        ValueError: In case of any invalid parameters.
    """

    def __init__(
        self, outputs: int, weights: np.ndarray = None, bias: np.ndarray = None
    ):
        """Initialize layer. See help(Dense) for more information."""
        if not (isinstance(outputs, int) and outputs > 0):
            raise ValueError("Argument outputs needs to be a positive integer.")

        if weights is not None and not isinstance(weights, np.ndarray):
            ValueError(
                "Argument weights should be None (for He-initialization)\
                 or np.ndarray with shape (k_x, k_y, channels_in, filters)."
            )

        if bias is not None and not isinstance(bias, np.ndarray):
            ValueError(
                "Argument bias should be None (for zero-initialization)\
                 or np.ndarray with shape (filters,)."
            )
        super().__init__()
        self.outputs = outputs
        self.weights = weights
        self.bias = bias
        self.optimizer = None

    def forward(self, data: np.ndarray) -> np.ndarray:
        """Forward pass through layer.

        Args:
            data (np.ndarray): Input with shape (batch, width, height, channels).

        Returns:
            np.ndarray: Output of the layer with shape self.output_shape.

        Raises:
            ValueError: In case of any invalid parameters.
        """
        if not self.input_shape == data.shape[1:]:
            raise ValueError(
                f"Incorrect input shape. Expected {self.input_shape}, got {data.shape[1:]}"
            )
        self.data = data

        # (batch_size, width, height, channels) -> (batch_size, width*height*channels )
        data_reshaped = data.reshape(data.shape[0], -1)

        # b - batch_size    # f - features  # o - outputs
        output = (
            np.einsum("bf, fo -> bo", data_reshaped, self.weights, optimize=True)
            + self.bias
        )
        self.forward_done = True
        return output

    def backward(self, dldy: np.ndarray) -> tuple:
        """Backward pass through the layer.

        Args:
            dldy (np.ndarray): Upstream gradient to propagate.

        Returns:
            tuple: Tuple of np.ndarray containing the downstream gradients
                for input, weights and bias respectively.

        Raises:
            RuntimeError: If layer has not forward propagated any input yet.
        """
        if not self.forward_done:
            raise RuntimeError(
                "Cannot backpropagate since no input has been forwarded through this layer yet.\
                 Make sure to propagate forward before doing backpropagation."
            )

        # (batch_size, width, height, channels) -> (batch_size, width*height*channels )
        data_reshaped = self.data.reshape(self.data.shape[0], -1)

        # b - batch_size    # f - features  # o - outputs
        dldx = np.einsum("bo, fo -> bf", dldy, self.weights, optimize=True)
        dldx = dldx.reshape(self.data.shape)
        dldw = np.einsum("bf, bo -> fo", data_reshaped, dldy, optimize=True)
        dldb = dldy.sum(axis=0)

        return dldx, dldw, dldb

    def update(self, grads: list) -> None:
        """Update the weights and bias.

        Args:
            grads (list): List of np.ndarray containing weights and bias gradients in that order.

        Returns:
            None

        Raises:
            RuntimeError: If optimizer not specified.
        """
        if self.optimizer is not None:
            self.weights, self.bias = self.optimizer.update(
                params=[self.weights, self.bias], grads=grads
            )
        else:
            raise RuntimeError(
                "Please specify layer optimizer before trying to update parameters."
            )

    def setup(self, input_shape: tuple, optimizer: Optimizer) -> None:
        """Setup the layer by specifying the input shape and initializing parameters.

        Args:
            input_shape (tuple): Tuple of integers specifying the shape of the input to the layer.
            optimizer (Optimizer): Optimizer to use for updating the layer's parameters.

        Returns:
            None
        """
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.init_params()

    def init_params(self) -> None:
        """Initialize parameters.

        Weights are initialized by He-initialization and bias with a zero vector.

        Returns:
            None
        """
        if self.weights is None:
            self.weights = np.random.randn(
                np.prod(self.input_shape), self.outputs
            ) / np.sqrt(np.prod(self.input_shape) / 2)
        if self.bias is None:
            self.bias = np.zeros(self.outputs)

    def __str__(self) -> str:
        """String representation of the layer."""
        n_params = np.size(self.weights) + np.size(self.bias)
        return f"{'Dense' : <20}{str(self.output_shape) : <20}{n_params : ^20}"

    @property
    def output_shape(self) -> tuple:
        """The layer's output shape without batch size."""
        return (self.outputs,)

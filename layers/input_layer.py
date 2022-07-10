"""This module contains the class InputLayer, the first layer in a neural network."""
import numpy as np

from layers.base_layer import Layer


class InputLayer(Layer):
    """Input layer is a placeholder for the input to the network.

    Attributes:
        input_shape (tuple): Tuple of integers specifying the shape of the input to the network.
            Batch size should not be included.

    """

    def __init__(self, input_shape: tuple):
        """Initialize layer. See help(InputLayer) for more information."""
        super().__init__()
        self.input_shape = input_shape

    def forward(self, data: np.ndarray) -> np.ndarray:
        """Forward pass through layer.

        Args:
            data (np.ndarray): Input with shape (batch, input_shape).

        Returns:
            np.ndarray: Output of the layer with shape self.output_shape.

        Raises:
            ValueError: In case of any invalid parameters.
        """
        if not self.input_shape == data.shape[1:]:
            raise ValueError(
                f"Incorrect input shape. Expected {self.input_shape}. Got {data.shape[1:]}"
            )
        self.forward_done = True
        return data

    def backward(self, dldy: np.ndarray) -> np.ndarray:
        """Backward pass through the layer.

        Args:
            dldy (np.ndarray): Upstream gradient to propagate.

        Returns:
            np.ndarray: The downstream gradient.

        Raises:
            RuntimeError: If layer has not forward propagated any input yet.
        """
        if not self.forward_done:
            raise RuntimeError(
                "Cannot backpropagate since no input has been forwarded through this layer yet.\
                    Make sure to propagate forward before doing backpropagation."
            )
        return dldy

    def __str__(self):
        """String representation of the layer."""
        return f"{'InputLayer' : <20}{str(self.output_shape) : <20}{0 : ^20}"

    @property
    def output_shape(self) -> tuple:
        """The layer's output shape without batch size."""
        return self.input_shape

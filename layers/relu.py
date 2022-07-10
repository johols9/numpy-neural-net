"""This module contains the class Relu, which in turn represents the Relu operation."""
import numpy as np

from layers.base_layer import Layer


class Relu(Layer):
    """Relu activation function layer. output = max(0, data)."""

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
        self.forward_done = True
        return np.maximum(0, data)

    def backward(self, dldy: np.ndarray) -> np.ndarray:
        """Backward pass through the layer.

        Args:
            dldy (np.ndarray): Upstream gradient to propagate.

        Returns:
            np.ndarray: Array containing the downstream gradient.

        Raises:
            RuntimeError: If layer has not forward propagated any input yet.
        """
        if not self.forward_done:
            raise RuntimeError(
                "Cannot backpropagate since no input has been forwarded through this layer yet.\
                Make sure to propagate forward before doing backpropagation."
            )
        return dldy * np.greater(self.data, 0)

    def setup(self, input_shape: np.ndarray) -> None:
        """Setup the layer by specifying the input shape.

        Args:
            input_shape (tuple): Tuple of integers specifying the shape of the input to the layer.

        Returns:
            None
        """
        self.input_shape = input_shape

    def __str__(self) -> str:
        """String representation of the layer."""
        return f"{'Relu' : <20}{str(self.output_shape) : <20}{0 : ^20}"

    @property
    def output_shape(self) -> tuple:
        """The layer's output shape without batch size."""
        return self.input_shape

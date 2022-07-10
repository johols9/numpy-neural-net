"""This module contains the class SoftmaxLoss, which is the output layer in a neural network."""
import numpy as np

from layers.base_layer import Layer


class SoftmaxLoss(Layer):
    """Softmax loss layer combining a softmax function with negative log likelihood loss."""

    def __init__(self):
        """Initialize layer. See help(SoftmaxLoss) for more information."""
        super().__init__()
        self.labels = None

    def forward(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Forward pass through layer.

        Args:
            data (np.ndarray): Input with shape (batch, outputs).
            labels (np.ndarray): Labels with shape (batch,). If None the loss is not calculated.

        Returns:
            np.ndarray: Softmax loss if labels is not None else None.

        Raises:
            ValueError: In case of any invalid parameters.
        """
        if not self.input_shape == data.shape[1:]:
            raise ValueError(
                f"Incorrect input shape. Expected {self.input_shape}, got {data.shape[1:]}"
            )
        self.data = data
        self.labels = labels

        loss = None
        if labels is not None:
            if not data.shape[0] == len(labels):
                raise ValueError(
                    f"Expected same amount of labels as batch size.\
                    Got {len(labels)} labels batch size {data.shape[0]}"
                )
            labels_idx = np.stack((np.arange(data.shape[0]), labels), axis=1)
            loss = -data[labels_idx[:, 0], labels_idx[:, 1]] + np.log(
                np.sum(np.exp(data), axis=1)
            )
            loss = np.mean(loss, keepdims=True)
        self.forward_done = True
        return loss

    def backward(self) -> np.ndarray:
        """Backward pass through the layer.

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

        labels_idx = np.stack((np.arange(len(self.labels)), self.labels), axis=1)
        dldx = np.exp(self.data) / np.sum(np.exp(self.data), axis=1, keepdims=True)
        dldx[labels_idx[:, 0], labels_idx[:, 1]] -= 1
        dldx /= self.data.shape[0]
        return dldx

    def setup(self, input_shape: tuple) -> None:
        """Setup the layer by specifying the input shape.

        Args:
            input_shape (tuple): Tuple of integers specifying the shape of the input to the layer.

        Returns:
            None
        """
        self.input_shape = input_shape

    def __str__(self) -> str:
        """String representation of the layer."""
        return f"{'SoftmaxLoss' : <20}{str(self.output_shape) : <20}{0 : ^20}"

    @property
    def output_shape(self):
        """Output shape of the layer."""
        return (1,)

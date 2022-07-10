"""This module contains the class Convolution2D, which represents a 2D convolution operation."""
import numpy as np

from layers.base_layer import Layer
from optimizers import Optimizer


class Convolution2D(Layer):
    """Convolutional layer for 2D-convolutions.

    Attributes:
        filters (int): Positive integer describing the number of kernels to slide over input.
        kernel_size (tuple): Tuple (k_x, k_y) of integers defining the shape of the kernel.
        padding (tuple, optional): Tuple of positive integers defining the padding to use.
            Defaults to (0,0), i.e., no padding.
        stride (tuple, optional): Tuple of positive integers defining the strides to use.
            Defaults to (1,1).
        weights (np.ndarray, optional): Weights matrix with shape (k_x, k_y, channels_in, filters).
            If None, weights are initalized with He-initialization.
        bias (np.ndarray, optional): Bias vector with shape (filters,).
            If None, bias is initialized as a zero vector.

    Raises:
        ValueError: In case of any invalid parameters.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        filters: int,
        kernel_size: tuple,
        padding: tuple = (0, 0),
        stride: tuple = (1, 1),
        weights: np.ndarray = None,
        bias: np.ndarray = None,
    ):
        """Initialize layer. See help(Convolution2D) for more information."""
        # pylint: disable=too-many-arguments
        if not (isinstance(filters, int) and filters > 0):
            raise ValueError("Argument filters must be a positive integer.")

        if not (
            isinstance(kernel_size, tuple)
            and len(kernel_size) == 2
            and isinstance(kernel_size[0], int)
            and isinstance(kernel_size[1], int)
            and kernel_size[0] > 0
            and kernel_size[1] > 0
        ):
            raise ValueError(
                "Argument kernel_size must be a tuple of integers with positive elements."
            )

        if not (
            isinstance(padding, tuple)
            and len(padding) == 2
            and isinstance(padding[0], int)
            and isinstance(padding[1], int)
            and padding[0] >= 0
            and padding[1] >= 0
        ):
            raise ValueError(
                "Argument padding must be a tuple of integers with positive elements."
            )

        if not (
            isinstance(stride, tuple)
            and len(stride) == 2
            and isinstance(stride[0], int)
            and isinstance(stride[1], int)
            and stride[0] > 0
            and stride[1] > 0
        ):
            raise ValueError(
                "Argument stride must be a tuple of integers with positive elements."
            )

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
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weights = weights
        self.bias = bias
        self.optimizer = None

        self.out_w = None
        self.out_h = None

    def _zero_padding(self, array: np.ndarray, padding: tuple) -> np.ndarray:
        """Zero pad the array.

        Args:
            array (np.ndarray): Array to pad with shape (batch size, width, height, channels).
            padding (tuple): tuple of integers defining the padding in each dimension.

        Returns:
            np.ndarray: Zero padded array.
        """
        batch_size, width, height, channels = array.shape
        p_w, p_h = padding[0], padding[1]
        new_width = width + 2 * p_w
        new_height = height + 2 * p_h
        padded_array = np.zeros((batch_size, new_width, new_height, channels))
        padded_array[:, p_w : new_width - p_w, p_h : new_height - p_h, :] = array
        return padded_array

    def _as_strided(self, array: np.ndarray) -> np.ndarray:
        """Create a view into the array representing a filter sliding over the array.

        Args:
            array (np.ndarray): Input array to create view into.

        Returns:
            np.ndarray: View into the array.
        """
        # pylint: disable=too-many-locals
        # pylint: disable=invalid-name
        bs_s, w_s, h_s, ch_s = array.strides
        bs, width, height, ch = array.shape

        kernel_width, kernel_height = self.kernel_size

        out_w = (width - kernel_width) // self.stride[0] + 1
        out_h = (height - kernel_height) // self.stride[1] + 1

        out_shape = (bs, out_w, out_h, kernel_width, kernel_height, ch)

        strides = (bs_s, self.stride[0] * w_s, self.stride[1] * h_s, w_s, h_s, ch_s)

        return np.lib.stride_tricks.as_strided(array, out_shape, strides)

    def forward(self, data: np.ndarray) -> np.ndarray:
        """Forward pass through layer.

        Args:
            data (np.ndarray): Input with shape (batch, width, height, channels).

        Returns:
            np.ndarray: Output of the layer with shape self.output_shape.

        Raises:
        ValueError: In case of invalid parameters.
        """
        if not self.input_shape == data.shape[1:]:
            raise ValueError(
                f"Incorrect input shape. Expected {self.input_shape}, got {data.shape[1:]}"
            )

        # Padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            data = self._zero_padding(data, self.padding)

        self.data = data

        input_view = self._as_strided(data)

        # b - batch_size    # w - out_width     # h - out_height    # i - kernel_width
        # j - kernel_height # c - channels      # f - nbr_filters
        output = (
            np.einsum("bwhijc, ijcf -> bwhf", input_view, self.weights, optimize=True)
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

        input_view = self._as_strided(self.data)

        # b - batch_size    # w - out_width     # h - out_height    # i - kernel_width
        # j - kernel_height # c - channels      # f - nbr_filters
        dldw = np.einsum("bwhijc, bwhf -> ijcf", input_view, dldy, optimize=True)
        dldb = dldy.sum(axis=(0, 1, 2))

        flipped_weights = self.weights[::-1, ::-1, :, :]
        padded_dldy = self._zero_padding(
            dldy, padding=(self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        )
        padded_dldy_view = self._as_strided(padded_dldy)

        # b - batch_size    # w - out_width     # h - out_height    # i - kernel_width
        # j - kernel_height # c - channels      # f - nbr_filters
        dldx = np.einsum(
            "bwhijc, ijcf -> bwhf", padded_dldy_view, flipped_weights, optimize=True
        )

        dldx = dldx[
            :,
            self.padding[0] : self.data.shape[1] - self.padding[0],
            self.padding[1] : self.data.shape[2] - self.padding[1],
            :,
        ]

        return dldx, dldw, dldb

    def update(self, grads: list) -> None:
        """Update the weights and bias.

        Args:
            grads (list): np.ndarray list containing the weights and bias gradients in that order.

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

        Returns:
            None
        """
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.out_w = (
            input_shape[0] + self.padding[0] * 2 - self.kernel_size[0]
        ) // self.stride[0] + 1
        self.out_h = (
            input_shape[1] + self.padding[1] * 2 - self.kernel_size[1]
        ) // self.stride[1] + 1
        self.init_params()

    def init_params(self) -> None:
        """Initialize parameters.

        Weights are initialized by He-initialization and bias with a zero vector.

        Returns:
            None
        """
        k_width, k_height = self.kernel_size
        channels = self.input_shape[-1]
        filters = self.filters
        if self.weights is None:
            self.weights = np.random.randn(
                k_width, k_height, channels, filters
            ) / np.sqrt(k_width * k_height * channels / 2)
        if self.bias is None:
            self.bias = np.zeros(filters)

    def __str__(self) -> str:
        """String representation of the layer."""
        n_params = np.size(self.weights) + np.size(self.bias)
        return f"{'Convolution2D' : <20}{str(self.output_shape) : <20}{n_params : ^20}"

    @property
    def output_shape(self) -> tuple:
        """The layer's output shape without batch size."""
        return (self.out_w, self.out_h, self.filters)

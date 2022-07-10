"""This module contains the class MaxPooling, which in turn performs a 2d-MaxPooling operation."""
import numpy as np

from layers.base_layer import Layer


class MaxPooling(Layer):
    """2D MaxPool layer.

    Attributes:
        pool_size (tuple, optional): Tuple of positive integers defining the pool shape.
            Defaults to (2,2).
        stride (tuple, optional): Tuple of positive integers defining the strides to use.
            Defaults to (2,2).

    Raises:
        ValueError: In case of any invalid parameters.
    """

    def __init__(self, pool_size: tuple = (2, 2), stride: tuple = (2, 2)):
        """Initialize layer. See help(MaxPooling) for more information."""
        if not (
            isinstance(pool_size, tuple)
            and len(pool_size) == 2
            and isinstance(pool_size[0], int)
            and isinstance(pool_size[1], int)
            and pool_size[0] > 0
            and pool_size[1] > 0
        ):
            raise ValueError(
                "Argument pool_size must be a tuple of integers with positive elements."
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
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.out_w = None
        self.out_h = None

    def _as_strided(self, array: np.ndarray) -> np.ndarray:
        """Create a view into the array representing the pool sliding over the input.

        Args:
            array (np.ndarray): Input array to create view into.

        Returns:
            np.ndarray: View into the array.
        """
        # pylint: disable=too-many-locals
        # pylint: disable=invalid-name
        bs_s, w_s, h_s, ch_s = array.strides
        bs, width, height, ch = array.shape

        pool_width, pool_height = self.pool_size

        out_w = (width - pool_width) // self.stride[0] + 1
        out_h = (height - pool_height) // self.stride[1] + 1

        out_shape = (bs, out_w, out_h, ch, pool_width, pool_height)

        strides = (bs_s, self.stride[0] * w_s, self.stride[1] * h_s, ch_s, w_s, h_s)

        return np.lib.stride_tricks.as_strided(array, out_shape, strides)

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
                f"Incorrect input shape. Expected {self.input_shape}. Got {data.shape[1:]}."
            )
        self.data = data

        input_view = self._as_strided(data)
        self.forward_done = True
        return np.amax(input_view, axis=(4, 5))

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
        input_view = self._as_strided(self.data)

        input_view_reshaped = input_view.reshape(
            input_view.shape[:4] + (self.pool_size[0] * self.pool_size[1],)
        )
        max_idxs = np.argmax(input_view_reshaped, axis=4)
        max_idxs = np.unravel_index(
            max_idxs.ravel(), shape=(self.pool_size[0], self.pool_size[1])
        )

        dldx = np.zeros(self.data.shape)
        dldx_view = self._as_strided(dldx)

        for out_shape, max_idx_r, max_idx_c, curr_dldy in zip(
            np.ndindex(dldy.shape), max_idxs[0], max_idxs[1], np.nditer(dldy.ravel())
        ):
            batch, out_w, out_h, channels = out_shape
            dldx_view[batch, out_w, out_h, channels, max_idx_r, max_idx_c] = curr_dldy

        return dldx

    def setup(self, input_shape: tuple) -> None:
        """Setup the layer by specifying the input shape.

        Args:
            input_shape (tuple): Tuple of integers specifying the shape of the input to the layer.

        Returns:
            None
        """
        self.input_shape = input_shape
        self.out_w = (input_shape[0] - self.pool_size[0]) // self.stride[0] + 1
        self.out_h = (input_shape[1] - self.pool_size[1]) // self.stride[1] + 1

    def __str__(self) -> str:
        """String representation of the layer."""
        return f"{'MaxPooling' : <20}{str(self.output_shape) : <20}{0 : ^20}"

    @property
    def output_shape(self) -> tuple:
        """The layer's output shape without batch size."""
        return (self.out_w, self.out_h, self.input_shape[2])

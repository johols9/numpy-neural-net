"""This module contains the abstract class Layer, which all other layers inherit from."""


class Layer:
    """Abstract Layer which all layers in the network inherit from."""

    def __init__(self):
        """Initialize layer. See help(Layer) for more information."""
        self.input_shape = None
        self.data = None
        self.forward_done = False

    def __str__(self):
        """String representation of the layer."""
        raise NotImplementedError

    @property
    def output_shape(self):
        """The layer's output shape without batch size."""
        return self.input_shape

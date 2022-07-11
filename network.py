"""Module containing the Network class which describes a neural network."""
import time
from copy import copy

import numpy as np

from layers.layers import InputLayer, Layer, SoftmaxLoss
from optimizers import Optimizer

ACCEPTED_LAYERS = {"Convolution2D", "Dense", "MaxPooling", "Relu", "SoftmaxLoss"}


class Network:
    """Neural network model.

    Attributes:
        optimizer (Optimizer): Optimizer to use when training the network.
        input_shape (tuple): Dimensions of the input data without batch size as a tuple of integers
            (width, height, channels). Ignored if model specified.
        model (str, optional): Path to saved model to load. If specified input_shape and batch_size
            will be ignored.

    Raises:
        ValueError: In case of any invalid parameters.
    """

    def __init__(
        self, optimizer: Optimizer = None, input_shape: tuple = None, model: str = None
    ):
        """Initialize network. See help(Network) for more information."""
        if model is not None:
            self.load_model(model)
        else:
            if input_shape is None or optimizer is None:
                raise ValueError(
                    "Please specify input_shape and optimizer if not loading a finished model."
                )
            if not (isinstance(input_shape, tuple) and len(input_shape) == 3):
                raise ValueError(
                    f"Incorrect value for input_shape. Expected tuple of shape\
                     (width, height, channels). Got {input_shape}"
                )
            if not isinstance(optimizer, Optimizer):
                raise ValueError(
                    f"Expected type {Optimizer.__name__} for optimizer.\
                     Got {type(optimizer).__name__}"
                )

            self.layers = [InputLayer(input_shape)]
            self.fitted = False
            self.batch_size = None
            self.optimizer = optimizer

    def add(self, layer: Layer) -> None:
        """Append a layer to the network.

        Args:
            layer (Layer): Layer to add last in the model.

        Returns:
            None

        Raises:
            TypeError: If Layer is not an accepted layer type.
        """
        if not type(layer).__name__ in ACCEPTED_LAYERS:
            raise TypeError(
                f"Incorrect layer type. Expected one of {ACCEPTED_LAYERS}.\
                 Got '{type(layer).__name__}'"
            )

        if hasattr(layer, "update"):
            layer.setup(self.layers[-1].output_shape, copy(self.optimizer))
        else:
            layer.setup(self.layers[-1].output_shape)
        self.layers.append(layer)

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        epochs: int,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        """Fit model to the data.

        Args:
            data (np.ndarray): Training data with shape (# train samples, width, height, channels).
            labels (np.ndarray): Labels corresponding to data with shape (# traing samples, ).
            val_data (np.ndarray): Validation data with shape
                (# validation samples, width, height, channels).
            val_labels (np.ndarray): Labels corresponding to validation data with shape
                (# validation samples, ).
            epochs (int): Positive integer specifying the number of epochs to perform.
            batch_size (int): Number of samples to train and evaluate at a time.
            shuffle (bool, optional): If True the training data is shuffled. Defaults to True.

        Returns:
            None

        Raises:
            ValueError: In case of invalid parameters.
            RunTimeError: If network has not been compiled yet.
        """
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        if not (
            isinstance(data, np.ndarray)
            and isinstance(labels, np.ndarray)
            and isinstance(val_data, np.ndarray)
            and isinstance(val_labels, np.ndarray)
        ):
            raise TypeError(
                "Input and labels must be of type np.ndarray for both training and validation data."
            )
        if not (isinstance(epochs, int) and isinstance(batch_size, int)):
            raise TypeError(
                "Arguments epochs and batch size must both be positive integers."
            )
        if not isinstance(shuffle, bool):
            raise TypeError(
                f"Argument shuffle must be a bool. Got {type(shuffle).__name__}."
            )

        if epochs <= 0:
            raise ValueError(
                f"Argument epochs must be a positive integer. Got {epochs}."
            )
        if batch_size <= 0:
            raise ValueError(
                f"Argument batch_size must be a positive integer. Got {batch_size}."
            )

        if not data.shape[0] == labels.shape[0]:
            raise ValueError(
                f"Number of training samples and number of training labels does not match.\
                 Got {data.shape[0]} training samples and  {labels.shape[0]} labels."
            )
        if not val_data.shape[0] == val_labels.shape[0]:
            raise ValueError(
                f"Number of validation samples and number of validation labels does not match.\
                 Got {val_data.shape[0]} training samples and  {val_labels.shape[0]} labels."
            )

        if not data.shape[1:] == self.input_shape:
            raise ValueError(
                f"Input has wrong dimensions. Expected {self.input_shape}. Got {data.shape[1:]}."
            )
        if not val_data.shape[1:] == self.input_shape:
            raise ValueError(
                f"Argument val_input has wrong dimensions. Expected {self.input_shape}.\
                 Got {val_data.shape[1:]}."
            )
        self.fitted = True
        self.batch_size = batch_size
        n_training = data.shape[0]

        indices = np.arange(n_training)
        for epoch in range(epochs):
            accuracy = 0
            loss = 0

            # Shuffle dataset
            if shuffle:
                np.random.shuffle(indices)

            batches = np.array_split(
                indices,
                indices_or_sections=np.arange(batch_size, n_training, batch_size),
            )

            start_time = time.time()

            for batch in batches:
                data_batch = data[batch]
                labels_batch = labels[batch]

                # Forward
                out, mean_batch_loss = self._forward(data_batch, labels_batch)
                loss += mean_batch_loss

                # Backward and optimization
                self._backward()

                pred_batch = np.argmax(out, axis=1)
                accuracy += np.sum(pred_batch == labels_batch)

            accuracy /= n_training
            loss /= len(batches)

            end_time = time.time()
            train_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

            # Validation set accuracy
            pred_val = self.predict(val_data)
            val_accuracy = np.mean(pred_val == val_labels)

            print(
                f"epoch: {epoch+1}/{epochs} \n \
                    time: {train_time} \n \
                    loss: {loss[0]} \n \
                    accuracy: {accuracy} \n \
                    validation accuracy: {val_accuracy} \n"
            )

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict input.

        Args:
            data (np.ndarray): Array with shape (samples, width, height, channels).

        Returns:
            np.ndarray: List of predictions.

        Raises:
            ValueError: In case of invalid parameters.
            RuntimeError: If the model is not fitted before trying to predict.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Input must be of type np.ndarray. Got {type(data).__name__}"
            )
        if not data.shape[1:] == self.input_shape:
            raise ValueError(
                f"input has wrong dimensions. Expected {self.input_shape}. Got {data.shape[1:]}."
            )
        if not self.fitted:
            raise RuntimeError(
                "Network has not been trained yet. Please run model.fit() before trying to predict."
            )

        batch_size = self.batch_size
        n_input = data.shape[0]
        batches = np.array_split(
            range(n_input),
            indices_or_sections=np.arange(batch_size, n_input, batch_size),
        )
        pred = []

        for batch in batches:
            data_batch = data[batch]
            out_batch = data_batch
            out_batch, _ = self._forward(data_batch)

            pred.append(np.argmax(out_batch, axis=1))
        return np.concatenate(pred)

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Get model accuracy on given input.

        Args:
            data (np.ndarray): Array to predict with shape (samples, width, height, channels).
            labels (np.ndarray): Labels or ground truth values.

        Returns:
            float: Accuracy of the model on the given input.

        Raises:
            ValueError: In case of invalid parameters.
            RuntimeError: If the model is not fitted before trying to evaluate.
        """
        pred = self.predict(data)
        accuracy = np.mean(pred == labels)
        return accuracy

    def summary(self) -> None:
        """Print a summary of the model.

        Returns:
            None
        """
        print(f"\n{'Layer' : <20}{'Output shape' : <20}{'# Parameters' : ^20}")
        print("=" * 60)
        for layer in self.layers:
            print(layer)
        print("=" * 60 + "\n\n")

    def save_model(self, path: str) -> None:
        """Save model as binary (.npy) file to path.

        Args:
            path (str): Location including filename to save model to.

        Returns:
            None

        Raises:
            ValueError: If fails to save model.
        """
        try:
            np.save(path, self.__dict__)
        except (TypeError, FileNotFoundError) as err:
            raise ValueError("Could not save model.") from err

    def load_model(self, path: str) -> None:
        """Load model from binary (.npy) file located at path.

        Args:
            path (str): Location including filename to load model from.

        Returns:
            None

        Raises:
            ValueError: If fails to load model.
        """
        try:
            model_dict = np.load(path, allow_pickle=True).item()
            self.__dict__.clear()
            self.__dict__.update(model_dict)
        except (TypeError, FileNotFoundError) as err:
            raise ValueError("Could not load model.") from err

    def _forward(self, data: np.ndarray, labels: np.ndarray = None) -> tuple:
        """Forward propagation through the whole network.

        Args:
            data (np.ndarray): Input with shape (samples, width, height, channels).
            labels (np.ndarray, optional): Labels with shape (samples,). If None,
                loss is not calculated.

        Returns:
            tuple: (out, loss), output from the 2nd to last layer and loss for the network.
        """
        # pylint: disable=too-many-function-args
        out = data
        loss = None
        for layer in self.layers[:-1]:
            out = layer.forward(out)
        layer = self.layers[-1]
        if isinstance(layer, SoftmaxLoss):
            loss = layer.forward(out, labels)
        return out, loss

    def _backward(self) -> None:
        """Backpropagation through the whole network.

        Returns:
            None
        """
        # pylint: disable=no-value-for-parameter
        grad = self.layers[self.nbr_layers - 1].backward()
        for idx in range(self.nbr_layers - 2, -1, -1):
            layer = self.layers[idx]
            if hasattr(layer, "update"):
                grad, weight_grads, bias_grads = layer.backward(grad)
                layer.update(grads=[weight_grads, bias_grads])
            else:
                grad = layer.backward(grad)

    @property
    def nbr_layers(self):
        """Number of layers in network."""
        return len(self.layers)

    @property
    def input_shape(self):
        """Shape of the input to the network."""
        return self.layers[0].input_shape

import torch
from ncps.torch import LTC
from ncps.wirings import FullyConnected


class LTCModel(LTC):
    """LTC-based force prediction model.
    
    Parameters
    ----------
    d_input : int
        Number of input features
    d_output : int, optional
        Output dimension for force (default: 3)
    d_hidden : int, optional
        Dimension of the hidden layer (default: 64)
    """

    def __init__(
        self,
        d_input,
        d_output=3,
        d_hidden=64,
    ):
        """Initialize the LTC model."""
        super().__init__(
            d_input,
            FullyConnected(
                d_hidden,
                d_output),
            return_sequences=False,
            batch_first=True,
        )

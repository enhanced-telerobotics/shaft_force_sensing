#!/bin/bash

# Run the training program
python -m shaft_force_sensing.training.trainer "$@"

# Then run the prediction program
python -m shaft_force_sensing.training.predictor "$@"
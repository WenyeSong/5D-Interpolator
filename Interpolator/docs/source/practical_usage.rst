Usage
=====

This section provides practical examples of how to use the 5D Interpolator system,
including the required dataset format and an example prediction workflow.

Dataset Format
--------------
The system expects datasets to be provided as a pickle (`.pkl`) file containing a
Python dictionary with the following structure:

- ``X``: a NumPy array of shape ``(N, 5)``, where each row represents a 5D input
- ``y``: a NumPy array of shape ``(N,)``, representing target values

Both ``X`` and ``y`` must contain numeric values and have matching lengths.

Example Dataset
---------------
An example dataset structure is shown below:

::

    {
        "X": [
            [0.12, -0.34, 0.56, 0.78, -0.90],
            [0.25,  0.10, 0.40, 0.15, -0.05],
            [0.24,  0.15, 0.50, 0.15, -0.06],
        ],
        "y": [1.23, 0.83. 0.87]
    }

The dataset can be created using NumPy and saved as a `.pkl` file using Python's
``pickle`` module.】


Train and Prediction Example
------------------------------

After uploading a valid dataset, choose parameters and click the **Train** button in the web interface. For example, the parameters could be:

- Layers: ``64,32,16``
- Learning rate: ``0.01``
- Epochs: ``200``

During training, the interface displays training progress, for example:

::

    Epoch 0:   Train=4.17, Val=4.35
    Epoch 20:  Train=3.18, Val=3.27
    ...
    Epoch 180: Train=0.04, Val=0.04

Once training is complete, a confirmation message is shown and the trained model
is saved on the backend.

Prediction Example
~~~~~~~~~~~~~~~~~~
After training, users can provide a single 5D input using the sliders or numeric
input fields. For example, setting the inputs to:

::

    X1 = 1.0
    X2 = 2.0
    X3 = 3.0
    X4 = 4.0
    X5 = 5.0

and clicking the **Predict** button will send the input to the backend and return
a predicted output value, such as:

::

    Prediction: 2.8473

The predicted value is displayed directly on the interface.




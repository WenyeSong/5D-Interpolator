User Guide
==========

This section describes how to use the 5D Interpolator system through its web-based
user interface. The system allows users to upload a dataset, train a configurable
neural network model, and perform predictions on new 5D inputs.

Overview
--------
The application follows a simple three-step workflow:

1. Upload a dataset containing 5D input features and corresponding targets
2. Train a neural network regression model with configurable parameters
3. Generate predictions for new 5D inputs through an interactive interface

Accessing the Interface
-----------------------
After successfully starting both the backend and frontend services, open a web
browser and navigate to:

::

    http://localhost:3000

You will see the main page titled **"5D Interpolator"**, which contains three
sections corresponding to the workflow steps.

Step 1: Upload Dataset
-----------------------
In the **Upload Dataset** section, click the file selection button and choose a
dataset file in `.pkl` format.

The uploaded file must contain a Python dictionary with the following keys:

- ``X``: a NumPy array of shape ``(N, 5)``, representing 5D input features
- ``y``: a NumPy array of shape ``(N,)``, representing target values

After clicking the **Upload** button, the backend validates the dataset format and
displays a summary on the interface, including:

- Number of data points
- Number of input features
- Target value range
- A preview of the first five input samples

If the dataset format is invalid, an error message will be displayed.

Step 2: Train Model
-------------------
Once a dataset has been successfully uploaded, users can proceed to the
**Train Model** section.

The following training parameters can be configured:

- **Layers**: a comma-separated list of hidden layer sizes
  (e.g., ``64,32,16``)
- **LR**: learning rate for the optimizer
- **Epochs**: number of training epochs

Click the **Train** button to start training. During training, the button is
temporarily disabled to prevent multiple submissions.

Training progress is displayed in real time, showing training and validation
loss values at regular intervals. Once training is complete, a confirmation
message is shown and the trained model is saved on the backend.

To ensure stable training and avoid unnecessary overfitting, the system applies
an early stopping mechanism during training. If the validation loss does not
improve for a predefined number of epochs, training will be stopped
automatically, even if a large number of epochs (e.g. ``1000``) is specified.

This prevents excessively long training runs and helps select a model that
generalises better to unseen data.


Step 3: Predict
---------------
After the model has been trained, users can generate predictions in the
**Predict** section.

Five input values (``X1`` to ``X5``) can be adjusted using sliders or numeric
input fields. Each input value ranges between ``-1`` and ``1``.

After specifying the input values, click the **Predict** button. The system sends
the input to the backend, performs inference using the trained model, and displays
the predicted output value on the interface.

Notes
-----
- A dataset must be uploaded before training can begin.
- The model must be trained before predictions can be made.
- All interactions between the frontend and backend are handled via RESTful API
  endpoints.

.. API Reference
.. =============

.. This section documents the backend REST API exposed by the FastAPI application.
.. All endpoints are automatically generated from the source code using Sphinx
.. autodoc.

.. Backend Endpoints
.. -----------------

.. .. automodule:: fivedreg.main
..    :members:
..    :undoc-members:
..    :show-inheritance:


API Reference
=============

This section documents the backend REST API exposed by the FastAPI application.

The backend provides endpoints for dataset upload, model training, prediction,
and performance benchmarking.

-------------------
REST API Endpoints
-------------------

Health Check
~~~~~~~~~~~~
**GET /health**

Check whether the backend service is running.

Response:
    - ``status``: string, should be ``"ok"``


Dataset Upload
~~~~~~~~~~~~~~
**POST /upload**

Upload a dataset in pickle format.

Expected file format:
    - Dictionary with keys ``X`` (N×5 array) and ``y`` (N array)

Response:
    - Dataset statistics and preview


Model Training
~~~~~~~~~~~~~~
**POST /train**

Train the neural network model with configurable hyperparameters.

Request body:
    - ``layers``: list of integers
    - ``lr``: float
    - ``epochs``: integer

Response:
    - Training logs
    - Early stopping information


Prediction
~~~~~~~~~~
**POST /predict**

Predict a scalar output for a single 5D input.

Request body:
    - ``X``: list of 5 floats

Response:
    - ``y_pred``: predicted scalar value


Performance Benchmark
~~~~~~~~~~~~~~~~~~~~~
**POST /benchmark**

Run performance and memory benchmarks for different dataset sizes.

Request body:
    - ``layers``
    - ``lr``
    - ``epochs``
    - ``dataset_sizes``

Response:
    - Training time
    - Peak memory usage
    - MSE and R² metrics


-------------------------
Request Schema Reference
-------------------------

.. automodule:: fivedreg.main
   :members: TrainRequest, PredictRequest, BenchmarkRequest
   :undoc-members:

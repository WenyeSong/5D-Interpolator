Testing
=======

This section describes the testing and evaluation strategy used to validate the
correctness, robustness, and performance characteristics of the system.

The testing suite combines automated unit and integration tests for functional
verification with dedicated profiling scripts for performance and scalability
analysis.

API Unit and Integration Tests
------------------------------

The backend API is tested using FastAPI’s built-in ``TestClient`` to simulate
HTTP requests without launching an external server. Tests are implemented using
``pytest`` and can be executed locally within the backend virtual environment.

The API tests cover the complete inference workflow, including health checks,
dataset upload, model training, and prediction.

Health Check
~~~~~~~~~~~~

A basic smoke test verifies that the backend service is running correctly by
issuing a ``GET /health`` request. The test asserts that the endpoint responds
with a valid status code and returns the expected JSON payload, confirming that
the application has been successfully initialised.

Dataset Upload and Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset upload and training pipeline is validated using a temporary
synthetic dataset generated during the test runtime. The test verifies that:

- The uploaded file is a valid ``.pkl`` file.
- The input feature matrix has the expected 5-dimensional shape.
- The ``POST /upload`` endpoint accepts the dataset and returns correct metadata.
- The ``POST /train`` endpoint successfully initiates the training process with
  user-specified hyperparameters.

Short training runs with a limited number of epochs are used to keep tests fast
while still exercising the full training logic, including validation and early
stopping behaviour.

Prediction Endpoint
~~~~~~~~~~~~~~~~~~~

After training, the prediction endpoint is tested by submitting a single 5D
input vector to the ``POST /predict`` endpoint. The test asserts that:

- The trained model is correctly loaded from disk.
- The endpoint returns a valid response.
- The output contains a scalar prediction value under the key ``y_pred``.

This confirms that the trained model can be reused for inference and that the
end-to-end pipeline from training to prediction is functional.

Error Handling and Robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional tests verify the robustness of the API under invalid or incomplete
inputs. These include:

- Rejecting datasets with incorrect feature dimensionality.
- Ensuring prediction requests fail gracefully if no trained model is available.

These checks confirm that the backend enforces input validation and provides
clear error responses for unsupported requests.

Performance and Profiling
-------------------------

Beyond functional correctness, the project includes dedicated performance and
memory profiling scripts located in the ``experiment`` directory.

These scripts evaluate the system’s scalability by measuring:

- Training time as a function of dataset size.
- Peak memory usage during model training.
- Peak memory usage during prediction.
- Prediction accuracy using Mean Squared Error (MSE) and coefficient of
  determination (R²).

Synthetic datasets of increasing size are generated to provide controlled and
reproducible benchmarks. Together, the automated tests and profiling experiments
demonstrate that the system behaves correctly, remains robust to invalid inputs,
and scales predictably as the workload increases.

Running the Tests
-----------------

All backend tests can be executed locally using the following commands:

::

    cd backend
    source .venv/bin/activate
    pip install -e .
    pytest

This setup ensures that the backend package is correctly installed and that all
tests are run in a consistent and reproducible environment.

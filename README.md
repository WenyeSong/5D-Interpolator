# 5D Interpolator

This project implements a full-stack web-app of a 5D interpolation system consisting of a
FastAPI backend and a Next.js frontend, deploying using Docker. Users can upload a 5D dataset, train a
configurable neural network model, and perform real-time predictions through
a web interface.

---

## System Architecture

The application is split into two primary services orchestrated by Docker Compose:

| Service   | Technology                     | Port | Description |
|-----------|---------------------------------|------|-------------|
| backend   | Python (FastAPI, Uvicorn)       | 8000 | REST API for data processing, model training, and prediction |
| frontend  | Next.js (React)                 | 3000 | Web interface for interacting with the backend API |



### Backend API

The backend is implemented using **FastAPI** and provides the following
endpoints:

- `GET /health` — Health check
- `POST /upload` — Upload a `.pkl` dataset containing `X` (N×5) and `y`
- `POST /train` — Train the neural network with configurable hyperparameters
- `POST /predict` — Predict a scalar output for a given 5D input

The backend runs on **port 8000** by default.

---

### Frontend

The frontend is implemented using **Next.js** and provides a web interface for:

- Dataset upload
- Model training configuration
- Real-time prediction using sliders or numeric inputs

The frontend runs on **port 3000** by default.

---

## Environment Variables

### 1. Frontend Configuration

The frontend service requires an environment variable to know where the backend API is running.
This is handled automatically by Docker Compose during the build phase.

| Variable                  | Location                  | Default Value              | Description |
|---------------------------|---------------------------|----------------------------|-------------|
| NEXT_PUBLIC_BACKEND_URL   | docker-compose.yml (arg)  | http://backend:8000        | Backend API base URL used by the frontend. Uses Docker internal service name |

### 2. Backend Configuration

The backend service is configured to run on port `8000` inside its container,
which is exposed to the host machine.

| Variable | Location              | Default Value | Description |
|---------|-----------------------|---------------|-------------|
| PORT    | backend/Dockerfile    | 8000          | The internal port Uvicorn listens on |
| HOST    | CMD in Dockerfile     | 0.0.0.0       | Binds Uvicorn to all interfaces for container access |


---

## Local Development (With Docker)

### 1. Download the whole project code 

### 2. Build and Run the Services

Navigate to the project root directory (where `docker-compose.yml` is located) and run:

```bash
cd Interpolator
docker-compose up --build
```
This command will build the backend container and the frontend container, and launch both services on a shared Docker network.

### 3. Access the Application

Once the containers are running:

- **Frontend (User Interface):**  
  http://localhost:3000

- **Backend API (Health Check):**  
  http://localhost:8000/health

### 3. Stop the Services

To stop and remove the containers and network:

```bash
docker compose down
```


---
## Alternative 1: Local Development with a shell script 
In addition to Docker-based deployment, the project provides a convenience
script for local development:


Make the script executable (only required once):

```bash
chmod +x scripts/run_local.sh
```

Then start the local development environment:

```bash
./scripts/run_local.sh
  ```

This script launches both the backend (FastAPI) and frontend (Next.js)
using local dependencies. The project will be available at http://localhost:3000


---
## Alternative 2: Local Development (Without Docker)

### Backend

```bash
cd backend
pip install -e .
uvicorn fivedreg.main:app --reload
```

Backend will be available at: http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```
Frontend will be available at: http://localhost:3000



---
## 📚 Documentation

Comprehensive documentation is provided using **Sphinx** under the `docs/` directory,
including API reference, usage guides, performance analysis, and testing descriptions.

To build the documentation locally:

```bash
./scripts/build_docs.sh
```

The generated HTML documentation can be opened directly in a web browser.
After building the docs, open the following file in your browser:

file:///<absolute-path-to-project>/docs/build/html/index.html

For example:

file:///Users/yourname/interpolator/docs/build/html/index.html


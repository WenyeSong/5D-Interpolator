#!/bin/bash
set -e

echo "Starting local development environment..."

# -------------------------
# Backend setup
# -------------------------
cd backend
echo "Setting up backend environment..."

BACKEND_PORT=8000

# automatically kill in use backend

echo "Checking if port $BACKEND_PORT is in use..."
PID_ON_PORT=$(lsof -ti tcp:$BACKEND_PORT || true)

if [ -n "$PID_ON_PORT" ]; then
  echo "Port $BACKEND_PORT is in use by PID(s): $PID_ON_PORT"
  echo "Killing process(es)..."
  kill -9 $PID_ON_PORT || true
  sleep 1
else
  echo "Port $BACKEND_PORT is free."
fi

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

pip install --upgrade pip
pip install .

echo "Starting backend (FastAPI)..."
uvicorn fivedreg.main:app --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!

echo "Backend started with PID $BACKEND_PID"

# Ensure backend is stopped on exit
cleanup() {
  echo "Stopping backend..."
  kill $BACKEND_PID 2>/dev/null || true
}
trap cleanup EXIT  ## avoid zombie

# -------------------------
# Frontend setup
# -------------------------
cd ../frontend
echo "Starting frontend (Next.js)..."

if [ -d "/opt/homebrew/opt/node@20/bin" ]; then
  export PATH="/opt/homebrew/opt/node@20/bin:$PATH"
  echo "Using Node.js 20 from Homebrew"
fi

if [ ! -d "node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

npm run dev

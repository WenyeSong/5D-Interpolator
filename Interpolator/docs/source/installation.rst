Installation
============

This project consists of a FastAPI backend and a Next.js frontend.
Both components are designed to run locally on a standard development machine.

Requirements
------------
- Python >= 3.10
- Node.js >= 18
- npm
- pip

Backend Installation
--------------------
1. Navigate to the backend directory:

   cd backend

2. Create and activate a virtual environment (optional but recommended):

   python -m venv c1CourseW
   source c1CourseW/bin/activate

3. Install Python dependencies:

   pip install -r requirements.txt

4. Start the backend server:

   uvicorn main:app --host 0.0.0.0 --port 8000

The backend API will be available at:
http://127.0.0.1:8000

Frontend Installation
---------------------
1. Navigate to the frontend directory:

   cd frontend

2. Install frontend dependencies:

   npm install

3. Start the frontend development server:

   npm run dev

The web interface will be available at:
http://localhost:3000

Notes
-----
The backend and frontend must be running simultaneously for full functionality.

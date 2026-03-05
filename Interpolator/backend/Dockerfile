# backend/docker

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .

# copy all code
COPY . .   

RUN pip install --upgrade pip
RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "fivedreg.main:app", "--host", "0.0.0.0", "--port", "8000"]

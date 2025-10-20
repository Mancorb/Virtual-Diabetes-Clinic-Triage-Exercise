# Use slim version for smaller size
FROM python:3.13-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Checks if FastAPI /health endpoint is OK every 30s
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1


# Run the FastAPI server (replace ENTRYPOINT if necessary)
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

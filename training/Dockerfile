FROM python:3.9-slim

WORKDIR /app


# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ./training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY ./training/ .

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONPATH=/app

# Default command
# CMD ["python", "train_xgboost.py", "--output-dir", "/app/output"]
CMD  ["python", "main.py"]
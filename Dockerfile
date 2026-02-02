
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any exact dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Environment variable to ensure output is flushed
ENV PYTHONUNBUFFERED=1

# Command to run the application
# We use uvicorn directly. Host 0.0.0.0 is crucial for Docker networking.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

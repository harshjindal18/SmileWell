# Start with a heavier image that has more libraries preinstalled
FROM python:3.11-bullseye

# Install system dependencies (only what is needed)
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your application code
COPY . .

# Install dlib separately first (to avoid building with rest)
RUN pip install --upgrade pip
RUN pip install dlib

# Install remaining Python dependencies
RUN pip install -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Start your Flask app
CMD ["python", "app.py"]

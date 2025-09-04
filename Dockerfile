# Use Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    curl \
    netcat-openbsd \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Install Yarn globally
RUN npm install -g yarn

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
WORKDIR /app/rapidfireai/frontend
RUN yarn install

# Create necessary directories
RUN mkdir -p /app/mlruns /root/db /app/logs

# Set permissions
RUN chmod 755 /root/db

# Copy the Docker-optimized start script and make it executable
COPY start-docker.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 5002 8080 3000

# Set environment variables
ENV MLFLOW_PORT=5002
ENV MLFLOW_HOST=0.0.0.0
ENV FRONTEND_PORT=3000
ENV FRONTEND_HOST=0.0.0.0
ENV API_PORT=8080
ENV API_HOST=0.0.0.0

# Use the start script as entrypoint
ENTRYPOINT ["/app/start.sh"]
CMD ["start"] 

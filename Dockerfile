# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install Node.js 20
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy API requirements first
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy web package files and install deps
COPY web/package.json /app/web/package.json
COPY web/package-lock.json /app/web/package-lock.json
WORKDIR /app/web
RUN npm ci

# Copy rest of repo
WORKDIR /app
COPY . /app

# Build Next.js
WORKDIR /app/web
RUN npm run build

# Supervisor to run both processes
WORKDIR /app
RUN pip install --no-cache-dir supervisor
COPY supervisord.conf /etc/supervisord.conf

EXPOSE 8000
EXPOSE 3000

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisord.conf"]

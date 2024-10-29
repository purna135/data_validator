# Use Python 3.11
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wkhtmltopdf \
    xvfb \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p uploads outputs && \
    chmod -R 777 uploads outputs

# Create a wrapper script for wkhtmltopdf with xvfb
RUN echo '#!/bin/bash\nxvfb-run -a --server-args="-screen 0, 1024x768x24" /usr/bin/wkhtmltopdf $*' > /usr/local/bin/wkhtmltopdf.sh \
    && chmod a+x /usr/local/bin/wkhtmltopdf.sh

# Set environment variables
ENV WKHTMLTOPDF_CMD=/usr/local/bin/wkhtmltopdf.sh
ENV MALLOC_ARENA_MAX=2
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Create a non-root user
RUN useradd -m myuser
USER myuser

# Expose the port
EXPOSE 8080

# Command to run the application
CMD exec gunicorn --bind :$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    main:app
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ticketsmith/ ./ticketsmith/
COPY configs/ ./configs/

# Set python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command (overridden by Job args)
CMD ["python", "-m", "ticketsmith.train", "--help"]

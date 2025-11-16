# Use official python slim
FROM python:3.10-slim

# Install system deps needed by OpenCV and common packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip and install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port (HF maps container ports)
EXPOSE 7860

# Run Streamlit from src/main.py
CMD ["streamlit", "run", "src/main.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]

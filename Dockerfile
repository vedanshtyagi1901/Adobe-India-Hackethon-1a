# Use a specific Python version for reproducibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all necessary scripts
COPY process_pdfs.py .
COPY minillm.py .

# Copy only the quantized fine-tuned model (remove base MiniLM)
COPY hf-model/fine_tuned_quantized /app/hf-model/fine_tuned_quantized

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pdfplumber \
    torch \
    transformers

# Optimize torch for CPU and reduce memory usage
ENV TORCH_HOME=/app
ENV PYTORCH_QNNPACK=1 

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the main application
CMD ["python", "process_pdfs.py"]

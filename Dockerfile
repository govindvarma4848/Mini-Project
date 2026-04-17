FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files
COPY backend/ backend/
COPY dataset/ dataset/
COPY fine_tuned_lora_model/ fine_tuned_lora_model/

# Expose port and run the API
EXPOSE 7860
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]


FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Upgrade pip before installing dependencies
RUN python -m pip install --upgrade pip

# Install project dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy prediction service and trained model
COPY predict.py .
COPY models/credit_risk_model.bin models/credit_risk_model.bin

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

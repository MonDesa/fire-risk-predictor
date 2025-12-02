FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Run with hypercorn
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8000"]

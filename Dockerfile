FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fast_api.py .
COPY lr_model.pkl .
COPY cat_model.pkl .
EXPOSE 8000
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
# 0000 makes it a server,allows it to be accessed from outside the container
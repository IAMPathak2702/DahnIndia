FROM python:3.11-slim

WORKDIR /app

# Install deps first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the serving code + the already-trained model artifact go into the
# image. Training needs pandas + the raw CSV and is deliberately run OFFLINE
# (see train/train_model.py), not as part of the Docker build -- that keeps
# the image small and the build fast, and means a bad training run can never
# accidentally ship a broken image.
COPY app/ ./app/
COPY model_artifact/ ./model_artifact/

EXPOSE 8000

# uvicorn boots app.main:app -> the lifespan handler loads the model ONCE
# at process startup, not on every request.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

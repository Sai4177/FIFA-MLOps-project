FROM python:3.11-slim
WORKDIR /app

COPY requirements-cicd.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# application + the small package that the pickled model needs to be un-pickled
COPY src/ ./src/
COPY tier_logic.py app.py ./
COPY fifa_tier_model.pkl metrics.json ./

ENV PYTHONPATH=/app/src:/app
EXPOSE 5000
CMD ["python", "app.py"]

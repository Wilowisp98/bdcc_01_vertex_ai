FROM python:3.11

WORKDIR /app
COPY app/requirements.txt ./
RUN pip install -r requirements.txt
COPY app/ .
COPY loader/csv_loader.py ./loader/
COPY loader/vertex_ai_images_loader.py ./loader/
COPY datasets/ ./datasets/

CMD ["python", "main.py"]
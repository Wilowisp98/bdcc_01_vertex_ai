FROM python:3.11

COPY /app/requirements.txt . 
RUN pip install -r requirements.txt
COPY /app/. /app/.
COPY /loader/csv_loader.py /loader/csv_loader.py
COPY /loader/vertex_ai_images_loader.py /loader/vertex_ai_images_loader.py
COPY /datasets/. /datasets/.

CMD ["python", "/app/main.py"]

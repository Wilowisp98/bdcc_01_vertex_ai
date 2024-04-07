FROM python:3.11

COPY /app/requirements.txt . 
RUN pip install -r requirements.txt
COPY /app/. /app/.
COPY /app/loader/csv_loader.py /app/loader/csv_loader.py
COPY /app/loader/vertex_ai_images_loader.py /app/loader/vertex_ai_images_loader.py
COPY /app/loader/tf_model_loader.py /app/loader/tf_model_loader.py
COPY /datasets/. /datasets/.

CMD ["python", "/app/main.py"]

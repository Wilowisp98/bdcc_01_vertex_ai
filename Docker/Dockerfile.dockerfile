FROM python:3.11

WORKDIR ../app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
COPY ../loader/csv_loader.py 
COPY ../loader/vertex_ai_images_loader
COPY ../datasets/.

CMD [python, main.py]
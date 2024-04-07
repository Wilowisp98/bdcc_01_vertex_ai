# Imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import logging
from google.cloud import bigquery
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Project ID
PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT', 'project01-418209') 
logging.info('Google Cloud project is {}'.format(PROJECT))

# Storage Bucket
BUCKET_NAME = PROJECT + '.appspot.com'
logging.info('Initialising access to storage bucket {}'.format(BUCKET_NAME))
APP_BUCKET = storage.Client(project=PROJECT).bucket(BUCKET_NAME)

def upload_file_to_storage_bucket(file_path, fname, **kwargs):
    blob = storage.Blob(fname, APP_BUCKET)
    with open(file_path, 'rb') as f:
        blob.upload_from_file(f, **kwargs)
    blob.make_public()
    return True

def main_model_loader():
    # files = get_files()
    files_to_upload = [
        '../app/static/tflite/dict.txt',
        '../app/static/tflite/model.tflite'
    ]
    
    for file in files_to_upload:
        upload_file_to_storage_bucket(file, file.split("/")[-1])
    logging.info('Model configs uploaded to storage bucket {}'.format(BUCKET_NAME))
    return True

#if __name__ == '__main__':
#    main()
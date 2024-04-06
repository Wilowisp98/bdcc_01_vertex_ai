# Imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import io
import os
import tqdm
import flask
import logging
import requests
from google.cloud import bigquery
from google.cloud import storage
          
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Project ID
PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT', 'project01-418209') 
logging.info('Google Cloud project is {}'.format(PROJECT))

# Big Query Client
logging.info('Initialising BigQuery client')
BQ_CLIENT = bigquery.Client(project=PROJECT)

# Storage Bucket
BUCKET_NAME = PROJECT + '.appspot.com'
logging.info('Initialising access to storage bucket {}'.format(BUCKET_NAME))
APP_BUCKET = storage.Client(project=PROJECT).bucket(BUCKET_NAME)


def upload_file_to_storage_bucket(
    file,
    fname,
    bucket,
    **kwargs
):
    # Upload image onto Storage bucket
    blob = storage.Blob(fname, APP_BUCKET)
    blob.upload_from_file(file, blob, **kwargs)
    blob.make_public()
    return True



def upload_img_to_storage_bucket(url:str, fname: str, pbar=None):
    # Get the bytes content of the URL
    response = requests.get(url)

    # Read the response as bytes
    img = io.BytesIO()
    img.write(response.content)

    # Move image read pointer to initial byte
    img.seek(0)

    # Upload image onto Storage bucket
    upload_file_to_storage_bucket(img, fname, APP_BUCKET, content_type=fname.split('.')[-1])
    if pbar is not None: pbar.update(1)

    return dict(bucket=APP_BUCKET, filename=fname)


def get_imgs_from_descs(descs=[]):
    query = '''
        Select ImageId, Description
        FROM `vertex_dataset.image_labels`
        INNER JOIN `vertex_dataset.classes` USING (Label)
        WHERE LOWER(description) IN ("{}") 
    '''.format('", "'.join(descs).lower())
    results = BQ_CLIENT.query(query).result()
    logging.info('get_imgs_from_labels: results={}'.format(results.total_rows))
    df = results.to_dataframe()
    df.reset_index(inplace=True, drop=True)
    return df


def generate_csv(imgs_df):
    csv: pd.Series = imgs_df['Upload_Dir'].apply(lambda d: 'gs://{}/{},'.format(APP_BUCKET.name, d['filename']))
    csv = csv + imgs_df['Description']
    return '\n'.join(csv)


def main_vertex_loader():
    # Defining descriptions to upload data onto BLOB and use the VertexAI tool with
    descs = [
        'Airplane', 
        'Mammal',
        'Car',
        'Tree',
        'Flower',
        'Human eye',
        'Footwear',
        'Dog',
        'Furniture',
        'Bird'
    ]

    # Get the images for the corresponding descriptions
    imgs_df = get_imgs_from_descs(descs)
    # Defining URL + name of images to download the data from
    imgs_df['URL'] = 'https://storage.googleapis.com/bdcc_open_images_dataset/images/' + imgs_df['ImageId'] + '.jpg'
    imgs_df['fname'] = 'TrainingImgs/' + imgs_df['Description'].str.lower() + '_' + imgs_df.groupby(by=['Description'])['ImageId'].transform('cumcount').astype(str) + '.jpg'

    # Limiting the number of images to 150 per description
    imgs_df = imgs_df[imgs_df['fname'].str.split('_').str[-1].str.split('.').str[0].astype(int) < 150]

    # Upload all images onto Blob storage
    pbar = tqdm.tqdm(total=imgs_df.shape[0], desc='Uploading Images to Blob')
    imgs_df['Upload_Dir'] = imgs_df.apply(lambda row: upload_img_to_storage_bucket(*row[['URL', 'fname']], pbar), axis=1)

    # Generate a csv ready for vertexAI
    csv = generate_csv(imgs_df)
    # Upload csv to blob
    file = io.StringIO()
    file.write(csv)
    file.seek(0)

    upload_file_to_storage_bucket(file, 'imgs_metadata.csv', APP_BUCKET, content_type='csv')

#if __name__ == '__main__':
#    main()
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import os
import pandas as pd

client = bigquery.Client(project='project01-418209')

def get_files():
    current_directory = '/'.join(__file__.split('/')[:-1])
    datasets = os.path.join(current_directory, '../datasets')
    file_names_csv = os.listdir(datasets)
    # Getting the name of the file
    file_names = [os.path.splitext(f)[0].replace('-','_') for f in file_names_csv if f.endswith('.csv')]
    # Creating a dict with the name of the file and the path of the file
    files = {name: os.path.join(datasets, f) for name, f in zip(file_names, file_names_csv)}
    
    return files


def create_bq(files, dataset_name='vertex_dataset'):
    dataset_ref = client.dataset(dataset_name)
    # Deleting the dataset if it already exists
    try:
        dataset = client.get_dataset(dataset_ref) 
        print(f"Dataset {dataset_name} already exists, deleting it first...") 
        client.delete_dataset(dataset, delete_contents=True, not_found_ok=True)
    except NotFound:
        pass

    # Creating the dataset
    vertex_dataset = client.create_dataset(dataset_name)
    print(f"Dataset {dataset_name} created successfully")

    # Iterating over the files and creating the respective tables
    for file, file_path in files.items():
        table_ref = vertex_dataset.table(file)
        
        # Read file
        df = pd.read_csv(file_path)
        columns = list(df.columns)
        job_config = bigquery.job.LoadJobConfig()

        # Setting the schema of the table iterating over the columns
        job_config.schema = [
            bigquery.SchemaField(f'{column_name}', 'STRING') for column_name in columns
        ]
        job_config.source_format = bigquery.SourceFormat.CSV

        # Loading the data from the file to the table
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        print(f"Table {file} created successfully")

def main_csv_loader():
    # files = get_files()
    files_to_upload_to_bq = [
        'https://storage.googleapis.com/bdcc_open_images_dataset/data/classes.csv',
        'https://storage.googleapis.com/bdcc_open_images_dataset/data/image-labels.csv',
        'https://storage.googleapis.com/bdcc_open_images_dataset/data/relations.csv'
    ]
    files_to_upload_to_bq = {
        file.split('/')[-1].replace('.csv', '').replace('-', '_'): file for file in files_to_upload_to_bq
    }
    create_bq(files_to_upload_to_bq)

#if __name__ == '__main__':
#    main()
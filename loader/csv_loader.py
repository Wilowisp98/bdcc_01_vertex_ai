from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import os
import pandas as pd

client = bigquery.Client()

def get_files():
    current_directory = os.getcwd()
    datasets = os.path.join(current_directory, '../datasets')
    file_names_csv = os.listdir(datasets)
    # Getting the name of the file
    file_names = [os.path.splitext(f)[0].replace('-','_') for f in file_names_csv if f.endswith('.csv')]
    # Creating a dict with the name of the file and the path of the file
    files = {name: os.path.join(datasets, f) for name, f in zip(file_names, file_names_csv)}
    
    return files

def create_bq(files):
    dataset_name = 'vertex_dataset'
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
        # Possible improvement
        # Deleting the table if it already exists (not needed bc the dataset is deleted, but might be an improvement)
        #try:
        #    client.get_table(table_ref)
        #    print(f"Table {file} already exists, deleting it first...")
        #    client.delete_table(table_ref, not_found_ok=True)
        #except NotFound:
        #    pass
        df = pd.read_csv(file_path)
        columns = list(df.columns)
        job_config = bigquery.job.LoadJobConfig()
        # Setting the schema of the table iterating over the columns
        job_config.schema = [
            bigquery.SchemaField(f'{column_name}', 'STRING') for column_name in columns
        ]
        job_config.source_format = bigquery.SourceFormat.CSV
        # Loading the data from the file to the table
        with open(file_path, "rb") as source_file:
            job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
        job.result()
        print(f"Table {file} created successfully")
        # Debug 
        #results = client.query(f'SELECT * FROM {table_ref.dataset_id}.{table_ref.table_id} LIMIT 10').result()
        #df = results.to_dataframe()
        #df = df.drop(index=0)
        #df.reset_index(inplace=True, drop=True)
        #print(df)

def main():
    files = get_files()
    create_bq(files)

if __name__ == '__main__':
    main()
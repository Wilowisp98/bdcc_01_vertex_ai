# Imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import flask
import logging
import os
import tfmodel
from google.cloud import bigquery
from google.cloud import storage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT', 'project01-418209') 
logging.info('Google Cloud project is {}'.format(PROJECT))

# Initialisation
logging.info('Initialising app')
app = flask.Flask(__name__)

logging.info('Initialising BigQuery client')
BQ_CLIENT = bigquery.Client(project=PROJECT)

BUCKET_NAME = PROJECT + '.appspot.com'
logging.info('Initialising access to storage bucket {}'.format(BUCKET_NAME))
APP_BUCKET = storage.Client(project=PROJECT).bucket(BUCKET_NAME)

logging.info('Initialising TensorFlow classifier')
TF_CLASSIFIER = tfmodel.Model(
    app.root_path + "/static/tflite/model.tflite",
    app.root_path + "/static/tflite/dict.txt"
)
logging.info('Initialisation complete')

# Run queries and return the result as a dataframe
def run_query(query: str):
    query = query.format(project_id=PROJECT) if '{project_id}' in query else query
    results = BQ_CLIENT.query(query).result()
    logging.info('classes: results={}'.format(results.total_rows))
    df = results.to_dataframe()
    df.reset_index(inplace=True, drop=True)
    return df

def to_flask_render_template(df):
    return df.to_numpy().tolist()

# End-point implementation
@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/classes')
def classes():
    query = '''
        Select Description, COUNT(*) AS NumImages
        FROM `{project_id}.vertex_dataset.image_labels`
        JOIN `{project_id}.vertex_dataset.classes` USING(Label)
        GROUP BY Description
        ORDER BY NumImages DESC
    '''
    results_df = run_query(query)
    return flask.render_template('classes.html', results=to_flask_render_template(results_df))


@app.route('/relations')
def relations():
    query = '''
        Select relation, COUNT(ImageId) AS NumImages
        FROM `{project_id}.vertex_dataset.relations`
        GROUP BY relation
        ORDER BY relation
    '''
    results_df = run_query(query)
    return flask.render_template('relations.html', results=to_flask_render_template(results_df))


@app.route('/image_info')
def image_info():
    image_id = flask.request.args.get('image_id')
    # TODO
    return flask.render_template('not_implemented.html')


@app.route('/image_search')
def image_search():
    description = flask.request.args.get('description', default='')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)
    results_df = run_query(
        '''
        Select ImageId
        FROM `{project_id}.vertex_dataset.image_labels`
        INNER JOIN `{project_id}.vertex_dataset.classes` USING (Label)
        WHERE LOWER(description) LIKE '%{desc}%'
        LIMIT {image_limit}
        '''.format(project_id=PROJECT, desc=description.lower(), image_limit=image_limit)
    )
    return flask.render_template('image_search.html', description=description, results=results_df['ImageId'].to_list())


@app.route('/relation_search')
def relation_search():
    class1 = flask.request.args.get('class1', default='%')
    relation = flask.request.args.get('relation', default='%')
    class2 = flask.request.args.get('class2', default='%')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)

    results_df = run_query(
        '''
        SELECT ImageId, c1.description desc1, relation, c2.description desc2
        FROM `{project_id}.vertex_dataset.relations`
        INNER JOIN `vertex_dataset.classes` c1 ON c1.label = Label1
        INNER JOIN `vertex_dataset.classes` c2 ON c2.label = Label2
        WHERE LOWER(relation) LIKE '%{relation}%'
            AND LOWER(c1.description) LIKE '%{class1}%'
            AND LOWER(c2.description) LIKE '%{class2}%'
        LIMIT {image_limit}
        '''.format(project_id=PROJECT, relation=relation.lower(), image_limit=image_limit, class1=class1.lower(), class2=class2.lower())
    )
    return flask.render_template('relation_search.html', class1=class1, relation=relation, class2=class2, results=to_flask_render_template(results_df))


@app.route('/image_classify_classes')
def image_classify_classes():
    with open(app.root_path + "/static/tflite/dict.txt", 'r') as f:
        data = dict(results=sorted(list(f)))
        return flask.render_template('image_classify_classes.html', data=data)
 
@app.route('/image_classify', methods=['POST'])
def image_classify():
    files = flask.request.files.getlist('files')
    min_confidence = flask.request.form.get('min_confidence', default=0.25, type=float)
    results = []
    if len(files) > 1 or files[0].filename != '':
        for file in files:
            classifications = TF_CLASSIFIER.classify(file, min_confidence)
            blob = storage.Blob(file.filename, APP_BUCKET)
            blob.upload_from_file(file, blob, content_type=file.mimetype)
            blob.make_public()
            logging.info('image_classify: filename={} blob={} classifications={}'\
                .format(file.filename,blob.name,classifications))
            results.append(dict(bucket=APP_BUCKET,
                                filename=file.filename,
                                classifications=classifications))
    
    data = dict(bucket_name=APP_BUCKET.name, 
                min_confidence=min_confidence, 
                results=results)
    return flask.render_template('image_classify.html', data=data)



if __name__ == '__main__':
    # When invoked as a program.
    logging.info('Starting app')
    app.run(host='127.0.0.1', port=8080, debug=True)

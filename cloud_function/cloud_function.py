import requests
from PIL import Image
from io import BytesIO
import flask
import tensorflow as tf
import numpy as np
import os
from google.cloud import storage

class Model:
    def _init_(self, model_file, dict_file):
        client = storage.Client()
        bucket = client.bucket('project01-418209.appspot.com')

        # Download the model file
        blob = storage.Blob(model_file, bucket)
        model = blob.download_as_bytes()

        # Download the dictionary file
        blob = storage.Blob(dict_file, bucket)
        dict_content = blob.download_as_text()

        self.labels = [line.strip().replace('_', ' ') for line in dict_content.splitlines()]
        self.interpreter = tf.lite.Interpreter(model_content=model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = self.input_details[0]['dtype'] == np.float32
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def classify(self, file, min_confidence):
        with Image.open(file).convert('RGB').resize((self.width, self.height)) as img:
            input_data = np.expand_dims(img, axis=0)
            if self.floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            model_results = np.squeeze(output_data)
            top_categories = model_results.argsort()[::-1]
            results = []
            for i in top_categories:
                if self.floating_model:
                    confidence = float(model_results[i])
                else:
                    confidence = float(model_results[i] / 255.0)
                if min_confidence != None and confidence < min_confidence:
                    break
                results.append(dict(label=self.labels[i], confidence='%.2f'%confidence))
            return results
        

TF_CLASSIFIER = Model(
    'model.tflite',
    'dict.txt'
)

def classify_image(request):
    min_confidence = 0.25

    # Get the image URL from the 'url' parameter
    image_url = request.args.get('url')
    if image_url is None:
        return flask.jsonify({'error': 'No image URL provided'}), 400

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    classifications = TF_CLASSIFIER.classify(image, min_confidence)
    return flask.jsonify(classifications)
import requests
from PIL import Image
from io import BytesIO
import flask
import tensorflow as tf
import numpy as np

app = flask.Flask(name)

class Model:

    def init(self, model_file, dict_file):
        with open(dict_file, 'r') as f:
            self.labels = [line.strip().replace('_', ' ') for line in f.readlines()]
        self.interpreter = tf.lite.Interpreter(model_path=model_file)
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
    app.root_path + "/static/tflite/model.tflite",
    app.root_path + "/static/tflite/dict.txt"
)

@app.route('/classify', methods=['POST'])
def classify_image():
    min_confidence = 0.25

    data = flask.request.get_json()
    image_url = data['image_url']
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    classifications = TF_CLASSIFIER.classify(image, min_confidence)
    return flask.jsonify(classifications)
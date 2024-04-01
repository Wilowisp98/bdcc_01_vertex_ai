import os
import requests
from google.cloud import vision

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT', 'project01-418209')

# Set QUOTA environmental Variable so google cloud knows which project to apply the QUOTA on 
os.environ['GOOGLE_CLOUD_QUOTA_PROJECT'] = PROJECT

client_vision = vision.ImageAnnotatorClient()




def classify_img(img_url):
    """Detects labels in a image url."""
    # Upload image onto vision
    image = vision.Image()
    image.source.image_uri = img_url

    response = client_vision.label_detection(image=image)
    labels = response.label_annotations
    print("Labels:")

    output = []
    for label in labels:
        output.append({
            'label': label.description,
            'confidence': label.score
            }
        )
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return output

if __name__ == '__main__':
    img_id = '01054f0eadfb2d5a'
    img_url = f'https://storage.googleapis.com/bdcc_open_images_dataset/images/{img_id}.jpg'
    output = classify_img(img_url)
    print(output)

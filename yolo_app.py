from flask import Flask, render_template, request
import numpy as np
import base64
import cv2
import os

from ultralytics import YOLO
model = YOLO('yolov8n.pt')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):
    image = cv2.imdecode( np.asarray(bytearray(image_stream.read()), dtype=np.uint8) , cv2.IMREAD_COLOR)

    results = model.predict(image, classes=0, conf=0.5)
    for i, r in enumerate(results):
        im_bgr = r.plot(conf=False)

    return im_bgr

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):

            predicted_image = predict_on_image(file.stream)

            retval, buffer = cv2.imencode('.png', predicted_image)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

            file.stream.seek(0)

            return render_template('index.html', render_result=True,
                                    detection_img_data=detection_img_base64)

    return render_template('index.html')

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=9898, host='0.0.0.0')
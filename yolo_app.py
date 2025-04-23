from flask import Flask, render_template, request
import numpy as np
import base64
import cv2
import os

from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Load a YOLOv8n object detection model

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Allowed image formats

app = Flask(__name__)

def allowed_file(filename):
    # Check if uploaded file has an allowed extension
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):
    # Convert uploaded image stream (binary) into an OpenCV BGR image
    image = cv2.imdecode(np.asarray(bytearray(image_stream.read()), dtype=np.uint8), cv2.IMREAD_COLOR)

    # Run YOLO prediction on the image, filtering by class ID 0 and confidence threshold of 0.5
    results = model.predict(image, classes=0, conf=0.5)

    # Draw detection results on the image (BGR format); disable confidence text
    for i, r in enumerate(results):
        im_bgr = r.plot(conf=False)  # im_bgr is the output image with boxes drawn

    return im_bgr  # Return the image with detection overlays

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            predicted_image = predict_on_image(file.stream)  # Run detection on uploaded image

            # Encode image with OpenCV as PNG, then convert it to base64 for embedding in HTML
            retval, buffer = cv2.imencode('.png', predicted_image)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

            file.stream.seek(0)  # Reset stream pointer in case you need to reuse it (not essential here)

            return render_template('index.html', render_result=True,
                                   detection_img_data=detection_img_base64)

    return render_template('index.html')

if __name__ == '__main__':
    # Set Flask environment variable for development mode if not already set
    os.environ.setdefault('FLASK_ENV', 'development')
    
    # Run the Flask app on all network interfaces at port 9898
    app.run(debug=False, port=9898, host='0.0.0.0')
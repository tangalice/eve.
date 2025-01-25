import base64
import io
import torch
from PIL import Image
from flask import Flask, request, jsonify
from io import BytesIO

# Load YOLOv5 model (suitable for object detection)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the base64 image data from the POST request
        data = request.get_json()
        image_data = data['image']

        # Convert the base64 string to a PIL image
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(img_data))

        # Perform detection with YOLOv5
        results = model(img)

        # Get predictions in the format of (label, x, y, width, height)
        predictions = []
        for *xyxy, conf, cls in results.xywh[0]:  # Get coordinates for each object
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, xyxy)  # Extract coordinates
            width = x2 - x1
            height = y2 - y1
            predictions.append({
                'label': label,
                'x': x1,
                'y': y1,
                'width': width,
                'height': height
            })

        # Return the predictions in JSON format
        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)

import os
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from keras.models import load_model
import base64

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

app = Flask(__name__)

global output
sections = []

# Load the model and labels for the scanner
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

@app.route('/search', methods=['GET', 'POST'])
def search():
    global output
    global sections
    if request.method == 'POST':
        search_query = request.form['search']
        # Use the Groq API to process the search query
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "assistant",
                    "content": """
                        You are a factual and informational assistant. When given the name of a feminine product such as a pad or tampon brand, provide a 1-sentence description of the following: 
                        1. Beneficial ingredients included in the product if there are any. If not, skip this line. Description title: "Beneficial Ingredients"
                        2. Harmful ingredients included in the product. Description title: "Harmful Ingredients"
                        3. Potential side effects of using the product. Description title: "Potential Side Effects"
                        4. Suggested safer alternatives to the product. Description title: "Safer Alternatives"
                        5. A ranking on a scale from 1-5 of how safe it is to use, with 1 highly unsafe and 5 being mostly safe. Description title: "Ranking"
                        Do not use overly complex medical jargon, but still communicate the key information. 
                        Start with the name of the product, then a new line. 
                        Then each description in a bulleted list, separated by a new line. Use the description title to start each bullet. 
                    """
                },
                {
                    "role": "user",
                    "content": search_query,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        output = chat_completion.choices[0].message.content
        sections = [section.strip() for section in output.split('*') if section.strip()]

        print(output)
        # Save the output to a file
        with open("output.txt", "w") as f:
            f.write(output)
    
    return render_template('search.html', sections=sections)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/scanner', methods=['GET'])
def scanner():
    # Render the scanner page
    return render_template('scanner.html')

@app.route('/scanner/predict', methods=['POST'])
def scanner_predict():
    # Decode the Base64 image from the request
    data = request.json
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    # Decode the image
    image_bytes = base64.b64decode(image_data.split(",")[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the image
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Predict with the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Return the prediction result
    return jsonify({
        "class": class_name,
        "confidence": float(confidence_score)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5004)

import cv2
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps
import pytesseract

# Load the image classification model
image_model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Path to Tesseract executable (modify for Windows if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_frame(frame):
    """
    Preprocess the video frame by converting it to grayscale.
    """
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray_frame)
        return pil_image
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def extract_text_from_frame(frame):
    """
    Extract text from a single video frame using Tesseract.
    """
    try:
        processed_image = preprocess_frame(frame)
        if processed_image is None:
            return ""
        config = "--oem 1 --psm 6"
        text = pytesseract.image_to_string(processed_image, config=config)
        return text
    except Exception as e:
        print(f"Error: {e}")
        return ""

def get_image_prediction(frame):
    """
    Get the prediction from the image recognition model.
    """
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = image_model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:], confidence_score

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 's' to capture a frame and process, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        cv2.imshow("Webcam Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Capture and process frame
            print("Processing frame...")

            # Get image model prediction
            image_class, image_confidence = get_image_prediction(frame)
            print(f"Image Recognition Prediction: {image_class} (Confidence: {image_confidence*100:.2f}%)")

            # Extract text from the frame
            text = extract_text_from_frame(frame)
            print("Extracted Text:", text)

            # Combine results from both models
            if text:
                print("Text-based prediction:", text)
                # You can improve this logic further based on text matching or additional filtering.
                # For now, we'll just show the highest confidence result between both models.

            # Decide based on the highest confidence score
            if image_confidence > 0.5:  # You can adjust the threshold
                print(f"Using Image Model Prediction: {image_class}")
            else:
                print(f"Using Text Model Prediction: {text.strip() if text else 'No text detected'}")

        elif key == ord('q'):  # Quit
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

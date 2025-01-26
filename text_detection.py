import cv2
from PIL import Image, ImageOps
import pytesseract

# Path to Tesseract executable (modify for Windows if needed)
# Uncomment and modify for Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_frame(frame):
    """
    Preprocess the video frame by converting it to grayscale.
    Args:
        frame: A single frame from the video feed.
    Returns:
        Image object: Preprocessed grayscale image.
    """
    try:
        # Convert the frame (BGR format) to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert the NumPy array to a PIL Image
        pil_image = Image.fromarray(gray_frame)
        return pil_image
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def extract_text_from_frame(frame):
    """
    Extract text from a single video frame.
    Args:
        frame: A single frame from the video feed.
    Returns:
        str: Extracted text from the frame.
    """
    try:
        # Preprocess the frame
        processed_image = preprocess_frame(frame)
        if processed_image is None:
            return ""

        # Configure Tesseract for cursive/stylized text
        config = "--oem 1 --psm 6"

        # Extract text
        text = pytesseract.image_to_string(processed_image, config=config)
        return text
    except Exception as e:
        print(f"Error: {e}")
        return ""

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 's' to capture a frame and extract text, or 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Display the video feed
        cv2.imshow("Webcam Feed", frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Press 's' to capture and extract text
            print("Processing frame...")
            text = extract_text_from_frame(frame)
            print("Extracted Text:\n", text)

            # Save the extracted text to a file
            with open("webcam_output.txt", "w", encoding="utf-8") as file:
                file.write(text)
            print("Text saved to 'webcam_output.txt'.")

        elif key == ord('q'):  # Press 'q' to quit
            print("Exiting...")
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

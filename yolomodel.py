import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from PIL import Image

# Stage 1: YOLO for Product Detection
def detect_products(image_path, yolo_model):
    """Detect menstrual products in an image using YOLO."""
    # Load the image
    image = cv2.imread(image_path)
    results = yolo_model(image)

    # Extract bounding boxes
    detections = []
    for box in results[0].boxes:  # Iterate through detected objects
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((x1, y1, x2, y2))

    return detections, image

# Stage 2: CNN for Brand Classification
def classify_brand(cropped_img, cnn_model, transform):
    """Classify the brand of a cropped product image."""
    # Convert cropped image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    img_tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension

    # Predict the brand
    cnn_model.eval()
    with torch.no_grad():
        output = cnn_model(img_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Main function
def main(image_path):
    # Load YOLO model (for product detection)
    yolo_model = YOLO("yolov5s.pt")  # Replace with your trained YOLO model

    # Load CNN model (for brand classification)
    cnn_model = models.resnet18(pretrained=True)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 3)  # 3 classes: Pads, Tampons, Cups
    cnn_model.load_state_dict(torch.load("brand_classifier.pth"))  # Load trained model weights

    # Define image transformations for CNN
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Detect products
    detections, image = detect_products(image_path, yolo_model)

    # Dictionary to map class indices to brand names
    brand_names = {0: "Brand_A", 1: "Brand_B", 2: "Brand_C"}

    # Loop through detections and classify each product
    for i, (x1, y1, x2, y2) in enumerate(detections):
        # Crop the detected product
        cropped_img = image[y1:y2, x1:x2]

        # Classify the brand
        brand_idx = classify_brand(cropped_img, cnn_model, transform)
        brand_name = brand_names[brand_idx]

        # Display the result
        print(f"Product {i + 1}: {brand_name}")

if __name__ == "__main__":
    image_path = "product_image.jpg"  # Replace with the path to your test image
    main(image_path)

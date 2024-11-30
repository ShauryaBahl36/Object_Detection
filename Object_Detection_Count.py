import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import time

# Paths to the MobileNet-SSD model files
PROTOTXT_PATH = "deploy.prototxt"  # Path to the prototxt file
MODEL_PATH = "mobilenet_iter_73000.caffemodel"  # Path to the pre-trained weights

# Create an output directory to save processed images
OUTPUT_DIR = "/Users/shauryabahl/Desktop/Object_Detection/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """Main function to execute the Streamlit app."""
    # Title and description
    st.title("Object Detection with MobileNet-SSD")
    st.write("Upload an image, detect objects, and save the output with bounding boxes.")
    print(os.path.exists(PROTOTXT_PATH))  # Should return True
    4

    # File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV-compatible format
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        (h, w) = image_cv2.shape[:2]

        # Load the MobileNet-SSD model
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

        # Preprocess the image for the model
        blob = cv2.dnn.blobFromImage(image_cv2, scalefactor=0.007843, size=(300, 300), mean=127.5)
        net.setInput(blob)

        # Perform forward pass to get detections
        detections = net.forward()

        # Initialize variables for object counting
        object_count = 0
        confidence_threshold = 0.5  # Confidence threshold for detections

        # Loop over detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                object_count += 1

                # Get the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label on the image
                label = f"Object {object_count}: {confidence:.2f}"
                cv2.rectangle(image_cv2, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(image_cv2, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed image in the output folder
        timestamp = int(time.time())
        output_filename = f"output_image_{timestamp}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, image_cv2)

        # Convert image back to RGB format for displaying in Streamlit
        output_image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # Display the processed image with bounding boxes
        st.image(output_image_rgb, caption="Processed Image with Detected Objects", use_column_width=True)

        # Display object count and saved file information
        st.success(f"Total objects detected: {object_count}")
        st.info(f"Output saved to: `{output_path}`")


# Entry point for the script
if __name__ == "__main__":
    main()

from ultralytics import YOLO
from PIL import Image
import os

# Load a pretrained YOLOv3 model
model = YOLO('yolov3u.pt')

input_dir = './input'
output_dir = './output'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over all images in the input directory
for image_name in os.listdir(input_dir):
    # Perform object detection on an image using the model
    results = model.predict(os.path.join(input_dir, image_name))

    # Get the detection results for the first (and only) image
    result = results[0]

    # Get the bounding boxes for all detected objects
    boxes = result.boxes

    # Get the class values of the boxes
    classes = boxes.cls

    # Get the index of the 'person' class = 0
    # Filter the boxes to only include persons
    person_boxes = boxes[classes == 0]

    # Load the original image
    image = Image.open(os.path.join(input_dir, image_name))

    print(f'data: {person_boxes.data}')

    # Iterate over each detected person
    for i, box in enumerate(person_boxes.data):
        # Get the bounding box coordinates
        x1, y1, x2, y2 = box[:4].tolist()

        # Crop the person from the image
        cropped_image = image.crop((x1, y1, x2, y2))

        # Save the cropped image to the output directory
        output_path = f'{output_dir}/{os.path.splitext(image_name)[0]}_person_{i}.jpg'
        cropped_image.save(output_path)

        # Print the path to the saved image
        print(f'Saved image to {output_path}')

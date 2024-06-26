import os
import cv2
from ultralytics import YOLO
import pdf2image
import img2pdf
import numpy as np

# Define Required method
def detect_objects(model, image, conf):
    """ Output inference image with bounding boxes
    Args:
    - model: YOLO model
    - image: input image
    - conf: confidence score to detect
    Return: image with bounding boxes drawn
    """
    # Predict on image
    results = model.predict(source=image, conf=conf, iou=0.8)
    boxes = results[0].boxes

    # Get bounding boxes
    if len(boxes) == 0:
        return image

    # Draw bounding boxes
    for box in boxes:
        detection_class_conf = round(box.conf.item(), 2)
        cls = list(ENTITIES_COLORS)[int(box.cls)]
        # Get start and end points of the current box
        start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
        end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
        # Draw bounding box
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        image = cv2.rectangle(img=image, pt1=start_box, pt2=end_box, color=ENTITIES_COLORS[cls], thickness=line_thickness)
        # Draw label
        text = cls + " " + str(detection_class_conf)
        font_thickness = max(line_thickness - 1, 1)
        (text_w, text_h), _ = cv2.getTextSize(text=text, fontFace=2, fontScale=line_thickness/3, thickness=font_thickness)
        image = cv2.rectangle(img=image, pt1=(start_box[0], start_box[1] - text_h - BOX_PADDING*2),
                               pt2=(start_box[0] + text_w + BOX_PADDING * 2, start_box[1]),
                               color=ENTITIES_COLORS[cls], thickness=-1)
        start_text = (start_box[0] + BOX_PADDING, start_box[1] - BOX_PADDING)
        image = cv2.putText(img=image, text=text, org=start_text, fontFace=0, color=(255, 255, 255),
                             fontScale=line_thickness/3, thickness=font_thickness)

    return image

# Define entities with it's bounding box colors
ENTITIES_COLORS = {
    "Caption": (191, 100, 21),
    "Footnote": (2, 62, 115),
    "Formula": (140, 80, 58),
    "List-item": (168, 181, 69),
    "Page-footer": (2, 69, 84),
    "Page-header": (83, 115, 106),
    "Picture": (255, 72, 88),
    "Section-header": (0, 204, 192),
    "Table": (116, 127, 127),
    "Text": (0, 153, 221),
    "Title": (196, 51, 2)
}
BOX_PADDING = 3
models = ['layout-identifier-model']
pdfs = ['PMC6103250_00003','PMC544398_00000']
pdfName = pdfs[0]
conf = 0.5

# Get input PDF file path from user
pdf_path = f"documents/{pdfName}.pdf"

# Create output folder
output_folder = "Layout-Identification-output/"
os.makedirs(output_folder, exist_ok=True)

# Convert PDF to images
images = pdf2image.convert_from_path(pdf_path)

# Process each image and save the output for each model
for modelName in models:
    model_folder =output_folder
    # os.makedirs(model_folder, exist_ok=True)
    confModel = f'{(conf * 100)}Per_{modelName}'
    # Load model
    DETECTION_MODEL = YOLO(f"models/{modelName}.pt")

    total_images = len(images)
    processed_images = 0

    for i, pil_image in enumerate(images):
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        processed_image = detect_objects(DETECTION_MODEL, image, conf)

        output_path = os.path.join(model_folder, f"{confModel}_{pdfName}_{i}.jpg")
        cv2.imwrite(output_path, processed_image)
        processed_images += 1

        completion_percentage = (processed_images / total_images) * 100
        print(f"Model: {modelName}, Processed image: {output_path}, Completion: {completion_percentage:.2f}%")

    # After processing all images
    with open(f"{output_folder}{pdfName}.pdf", "wb") as f:
        img_paths = [os.path.join(model_folder, f"{confModel}_{pdfName}_{i}.jpg") for i in range(total_images)]
        f.write(img2pdf.convert(img_paths))

    # Delete the images
    for img_path in img_paths:
        os.remove(img_path)
    
    print(f"Model: {modelName}, Work completed!")

print("Done!")
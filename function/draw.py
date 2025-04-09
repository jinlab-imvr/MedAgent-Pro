import os
from PIL import Image, ImageDraw

def draw_bbox(image_path, bboxes, output_path):
    """
    Draw bounding boxes on an image and save the result.

    Args:
        image_path (str): Path to the input image.
        bboxes (list of tuples): List of bounding boxes, each defined as (x1, y1, x2, y2).
        output_path (str): Path to save the output image with bounding boxes.
    """
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw each bounding box
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Save the modified image
    image.save(output_path)
    return image
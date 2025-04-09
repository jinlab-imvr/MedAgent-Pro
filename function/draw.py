import os
from PIL import Image, ImageDraw
from skimage import io
import numpy as np

def draw_bbox(image_path, bbox, output_path):
    image = Image.open(image_path)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    x1, y1, x2, y2 = bbox
    x1 = x1 * width
    y1 = y1 * height
    x2 = x2 * width
    y2 = y2 * height

    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    image.save(output_path)
    return image

def draw_mask(image_path, mask_path, output_path):
    """
    mask: a 1-channel mask, value 0 is the background and 1 is the foreground
    """

    img_np = io.imread(image_path)
    mask = io.imread(mask_path)
    if img_np.ndim == 2:
        img_color = np.stack([img_np] * 3, axis=-1)
    elif img_np.ndim == 3 and img_np.shape[2] == 1:
        img_color = np.concatenate([img_np] * 3, axis=2)
    else:
        img_color = img_np.copy()
    overlay_color = np.array([255, 0, 0], dtype=img_color.dtype)
    alpha = 0.5
    overlayed_img = img_color.copy()
    foreground = (mask == 1)
    overlayed_img[foreground] = (
        alpha * overlay_color + (1 - alpha) * overlayed_img[foreground]
    ).astype(img_color.dtype)
    io.imsave(
        output_path,
        overlayed_img,
        check_contrast=False,
)
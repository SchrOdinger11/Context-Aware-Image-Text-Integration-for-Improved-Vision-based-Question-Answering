# %%
from models import NERModel, SamModelPrediction
import json
import cv2
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import os


# %%
file_path = '/Users/sudhanshu/Desktop/UMASS_COURSES_SEMESTERS/SEM_2/NLP/Project/dataset/combined_data.json'
ner = NERModel()
sam_predictor = SamModelPrediction()


def show_mask(mask, ax, random_color=False):
    """
    Adds a mask to an existing Axes object with the option to use a random color.

    Args:
    mask (array): A 2D numpy array representing the mask.
    ax (matplotlib.axes.Axes): The Axes object where the mask will be added.
    random_color (bool): If True, the mask will be displayed in a random color.
    """
    if random_color:
        # Generate a random color with transparency
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Use a specific color with transparency
        color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape
    # Multiply the mask by the color and reshape for proper display
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, interpolation='none', alpha=0.5)



# %%
def create_segmented_image(pil_image, masks, background_color=(0, 0, 0)):
    """
    Create a segmented image where only the areas defined by masks are visible,
    and the rest is set to a specified background color.

    Args:
    pil_image (PIL.Image): The original image.
    masks (list of numpy arrays): A list of 2D numpy arrays representing masks.
    background_color (tuple): RGBA color tuple for the background, defaults to transparent.

    Returns:
    PIL.Image: The segmented image with only masked areas from the original image.
    """
    
   
    if  len(masks) ==0:  # Check if the masks list is empty
        
        return pil_image
    # Convert the PIL Image to a NumPy array (RGBA for transparency handling)
    image_np = np.array(pil_image.convert('RGB'))
    
    # Prepare a blank canvas for the final image with the background color
    final_image_np = np.full(image_np.shape, background_color, dtype=np.uint8)

    # Process each mask
    for mask in masks:
        # Apply the mask to copy relevant parts from the original image to the final image
        final_image_np[mask] = image_np[mask]

    # Convert back to PIL Image
    segmented_image = Image.fromarray(final_image_np, 'RGB')
    return segmented_image





# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from imageio import imread, imwrite  # Replacing deprecated scipy.misc functions
from os import mkdir
from os.path import exists, join, basename, splitext
from models import generator

def generate(ir_path, vis_path, model_path, index, output_path='output/', scale_factor=255.0):
    # Load and normalize images
    try:
        ir_img = imread(ir_path) / scale_factor
        vis_img = imread(vis_path) / scale_factor
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Ensure the images have 3 dimensions (H, W, C)
    if len(ir_img.shape) == 2:
        ir_img = np.expand_dims(ir_img, axis=-1)
    if len(vis_img.shape) == 2:
        vis_img = np.expand_dims(vis_img, axis=-1)

    # Reshape images to add batch dimension
    ir_img = np.expand_dims(ir_img, axis=0)
    vis_img = np.expand_dims(vis_img, axis=0)

    # Initialize the generator model
    G = generator('Generator')

    @tf.function
    def generate_image(vis, ir):
        return G.transform(vis=vis, ir=ir)

    # Load model weights
    try:
        G.load_weights(model_path)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Generate the fused image
    try:
        output_image = generate_image(vis_img, ir_img)
        output_image = output_image[0, :, :, 0]  # Remove batch and channel dimensions

        # Create output directory if it doesn't exist
        if not exists(output_path):
            mkdir(output_path)

        # Save the output image
        output_image_path = join(output_path, f'{index}.bmp')
        imwrite(output_image_path, (output_image * scale_factor).astype(np.uint8))  # Convert back to 8-bit image

        print(f"Image saved at {output_image_path}")

    except Exception as e:
        print(f"Error during image generation: {e}")

# Example usage:
# generate('ir_image_path.bmp', 'vis_image_path.bmp', 'model_checkpoint_path', 0)

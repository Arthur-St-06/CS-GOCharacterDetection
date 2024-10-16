import cv2
import numpy as np
import os
import random

# Function to augment an image and update labels
def augment_image_and_labels(image_path, label_path, output_dir, num_augmentations):
    # Load the image and its corresponding YOLO labels
    image = cv2.imread(image_path)
    labels = []
    with open(label_path, 'r') as label_file:
        labels = label_file.read().strip().split('\n')

    # Get image dimensions
    height, width, _ = image.shape

    for i in range(num_augmentations):
        # Randomly choose an augmentation operation
        augmentation = random.choice(['flip', 'rotate', 'blur', 'noise'])

        # Create a copy of the original image and labels
        augmented_image = image.copy()
        augmented_labels = labels.copy()

        if augmentation == 'flip':
            augmented_image = cv2.flip(augmented_image, 1)  # Horizontal flip
            # Update labels for horizontal flip
            for i, label in enumerate(augmented_labels):
                values = label.split()
                x_center = float(values[1])
                x_center = 1.0 - x_center  # Flip x-coordinate
                augmented_labels[i] = f"{values[0]} {x_center} {values[2]} {values[3]} {values[4]}"

        elif augmentation == 'rotate':
            angle = random.uniform(-10, 10)  # Random rotation angle
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            augmented_image = cv2.warpAffine(augmented_image, matrix, (width, height))
            # Update labels for rotation
            for i, label in enumerate(augmented_labels):
                values = label.split()
                x_center = float(values[1])
                y_center = float(values[2])
                new_x_center = (x_center * np.cos(np.radians(angle))) - (y_center * np.sin(np.radians(angle)))
                new_y_center = (x_center * np.sin(np.radians(angle))) + (y_center * np.cos(np.radians(angle)))
                augmented_labels[i] = f"{values[0]} {new_x_center} {new_y_center} {values[3]} {values[4]}"

        # Save the augmented image and labels
        output_image_path = os.path.join(output_dir, f"augmented_{i}.jpg")
        output_label_path = os.path.join(output_dir, f"augmented_{i}.txt")

        cv2.imwrite(output_image_path, augmented_image)
        with open(output_label_path, 'w') as output_label_file:
            output_label_file.write('\n'.join(augmented_labels))

# Example usage
image_path = 'input.jpg'  # Path to the original image
label_path = 'input.txt'  # Path to the YOLO format label file for the original image
output_dir = 'augmented_data'  # Directory to save augmented images and labels
num_augmentations = 5  # Number of augmentations to create

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

augment_image_and_labels(image_path, label_path, output_dir, num_augmentations)

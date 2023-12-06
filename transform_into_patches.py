import os
from PIL import Image


def extract_patches(image_path):
    image = Image.open(image_path)
    width, height = image.size
    patch_size = 256
    num_patches_horizontal = width // patch_size
    num_patches_vertical = height // patch_size

    for i in range(num_patches_horizontal):
        for j in range(num_patches_vertical):
            x = i * patch_size
            y = j * patch_size
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            patch_number = i * num_patches_vertical + j
            new_image_path = f"{base_name}_new_{patch_number}.jpg"
            patch.save(new_image_path)


def process_source_images(directory_path):
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            extract_patches(file_path)


# Example usage:
process_source_images("/path/to/source/images")

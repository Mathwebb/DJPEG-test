import os

CASIA2_PATH = 'D:\data_sets\general_image_tampering\CASIA\CASIA2'
TARGET_PATH = 'D:\data_sets\double_jpeg_compression\casia_double_single'


def read_authentic_images(target_path):
    authentic_images = []
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                authentic_images.append(os.path.join(root, file))
    return authentic_images


def read_tampered_images(target_path):
    tampered_images = []
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                tampered_images.append(os.path.join(root, file))
    return tampered_images


def equalize_image_quantities(authentic_images, tampered_images):
    if len(authentic_images) > len(tampered_images):
        authentic_images = authentic_images[:len(tampered_images)]
    elif len(tampered_images) > len(authentic_images):
        tampered_images = tampered_images[:len(authentic_images)]
    return authentic_images, tampered_images


def separate_to_batches(authentic_images, tampered_images, total_batches):
    authentic_images_batches = []
    tampered_images_batches = []
    batch_size = int(len(authentic_images) / total_batches)
    for i in range(total_batches):
        authentic_images_batches.append(
            authentic_images[i * batch_size:(i + 1) * batch_size])
        tampered_images_batches.append(
            tampered_images[i * batch_size:(i + 1) * batch_size])
    return authentic_images_batches, tampered_images_batches


def create_single_double_dataset(authentic_images, tampered_images, batch_quantity):
    authentic_images, tampered_images = equalize_image_quantities(
        authentic_images, tampered_images)
    authentic_images_batches, tampered_images_batches = separate_to_batches(
        authentic_images, tampered_images, batch_quantity)
    return authentic_images_batches, tampered_images_batches


def main():
    authentic_images = read_authentic_images(
        os.path.join(CASIA2_PATH, 'authentic'))
    tampered_images = read_tampered_images(
        os.path.join(CASIA2_PATH, 'tampered'))
    authentic_images_batches, tampered_images_batches = create_single_double_dataset(
        authentic_images, tampered_images, 20)
    for i in range(1, 21):
        if not os.path.exists(os.path.join(TARGET_PATH, 'single', str(i))):
            os.mkdir(os.path.join(TARGET_PATH, 'single', str(i)))
            os.mkdir(os.path.join(TARGET_PATH, 'double', str(i)))
        for authentic_image, tampered_image in zip(authentic_images_batches[i - 1], tampered_images_batches[i - 1]):
            if not os.path.exists(os.path.join(TARGET_PATH, 'single', str(i), os.path.basename(authentic_image))):
                os.rename(authentic_image, os.path.join(
                    TARGET_PATH, 'single', str(i), os.path.basename(authentic_image)))
            if not os.path.exists(os.path.join(TARGET_PATH, 'double', str(i), os.path.basename(tampered_image))):
                os.rename(tampered_image, os.path.join(
                    TARGET_PATH, 'double', str(i), os.path.basename(tampered_image)))


if __name__ == '__main__':
    main()

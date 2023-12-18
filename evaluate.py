import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from application import djpeg

ORIGINAL_DIR = 'D:\\data_sets\\general_image_tampering\\CASIA\\CASIA1\\authentic'
TAMPERED_DIR = 'D:\\data_sets\\general_image_tampering\\CASIA\\CASIA1\\tampered'


def create_jpeg_masks(original_dir: str, tampered_dir: str):
    if not os.path.exists(os.path.join(original_dir, 'results')):
        os.mkdir(os.path.join(original_dir, 'results'))

    for file_name in os.listdir(original_dir):
        if file_name.endswith('.jpg') and not file_name.endswith('_result.jpg'):
            djpeg(os.path.join(original_dir, file_name))

    if not os.path.exists(os.path.join(tampered_dir, 'results')):
        os.mkdir(os.path.join(tampered_dir, 'results'))

    for file_name in os.listdir(tampered_dir):
        if file_name.endswith('.jpg') and not file_name.endswith('_result.jpg'):
            djpeg(os.path.join(tampered_dir, file_name))


def predict(image_path: str) -> str:
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    hist = cv.calcHist([image], [0], None, [256], (0, 255))
    count = 0
    cumulative_hist = []
    for intensity in hist:
        count += intensity[0]
        cumulative_hist.append(count)

    if cumulative_hist[127] >= 0.5 * cumulative_hist[255]:
        return 'tampered'
    else:
        return 'original'


def evaluate(original_results_dir: str, tampered_results_dir: str) -> dict:
    metrics: dict[str, float] = {
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    for file_name in os.listdir(original_results_dir):
        file_path = os.path.join(original_results_dir, file_name)
        if predict(file_path) == 'original':
            metrics['true_negatives'] += 1
        else:
            metrics['false_negatives'] += 1

    for file_name in os.listdir(tampered_results_dir):
        file_path = os.path.join(tampered_results_dir, file_name)
        if predict(os.path.join(file_path)) == 'tampered':
            metrics['true_positives'] += 1
        else:
            metrics['false_positives'] += 1

    for metric in metrics.values():
        print(metric)

    metrics['accuracy'] = (metrics['true_positives'] +
                           metrics['true_negatives']) / (metrics['true_positives'] +
                                                         metrics['true_negatives'] +
                                                         metrics['false_positives'] +
                                                         metrics['false_negatives'])
    metrics['precision'] = metrics['true_positives'] / \
        (metrics['true_positives'] + metrics['false_positives'])
    metrics['recall'] = metrics['true_positives'] / \
        (metrics['true_positives'] + metrics['false_negatives'])
    return metrics


if __name__ == '__main__':
    create_jpeg_masks(ORIGINAL_DIR, TAMPERED_DIR)

    original_results = [file_name for file_name in os.listdir(
        os.path.join(ORIGINAL_DIR, 'results')) if file_name.endswith('_result.jpg')]
    tampered_results = [file_name for file_name in os.listdir(
        TAMPERED_DIR) if file_name.endswith('_result.jpg')]

    metrics = evaluate(os.path.join(ORIGINAL_DIR, 'results'),
                       os.path.join(TAMPERED_DIR, 'results'))
    print(metrics)

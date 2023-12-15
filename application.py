import os
import torch
import torch.nn.functional as F
import math
from PIL import JpegImagePlugin
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from djpegnet import Djpegnet
from data import read_q_table


def _extract_patches(Y, patch_size, stride):
    patches = list()
    h, w = Y.shape[0:2]
    H = (h - patch_size) // stride
    W = (w - patch_size) // stride
    for i in range(0, H*stride, stride):
        for j in range(0, W*stride, stride):
            patch = Y[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return patches, H, W


def localizing_double_JPEG(Y, qvectors, stride=32, batch_size=32, net=None, device=None):
    net.eval()
    result = 0
    PATCH_SIZE = 256

    qvectors = torch.from_numpy(qvectors).float()
    qvectors = qvectors.to(device)
    qvectors = torch.unsqueeze(qvectors, axis=0)

    # result = np.zeros_like(Y)

    patches, H, W = _extract_patches(
        Y, patch_size=PATCH_SIZE, stride=stride)
    result = np.zeros((H, W))
    print('Number of patches: {}'.format(len(patches)))
    print('Number of batches: {}'.format(
        math.ceil(len(patches) / batch_size)))
    print('Patch size: {}'.format(PATCH_SIZE))

    # import pdb; pdb.set_trace()
    num_batches = math.ceil(len(patches) / batch_size)

    result_flatten = np.zeros((H*W))
    for i in range(num_batches):
        print('[{} / {}] Detecting...'.format(i, num_batches))
        if i == (num_batches-1):  # last batch
            batch_Y = patches[i*batch_size:]
        else:
            batch_Y = patches[i*batch_size:(i+1)*batch_size]

        batch_size = len(batch_Y)
        batch_Y = np.array(batch_Y)
        batch_Y = torch.unsqueeze(torch.from_numpy(
            batch_Y).float().to(device), axis=1)
        batch_qvectors = torch.repeat_interleave(qvectors, batch_size, dim=0)
        batch_output = net(batch_Y, batch_qvectors)
        batch_output = F.softmax(batch_output, dim=1)

        result_flatten[(i*batch_size):(i*batch_size)+batch_size] = \
            batch_output.detach().cpu().numpy()[:, 0]

    result = np.reshape(result_flatten, (H, W))

    return result


def djpeg(file_path: str, stride=32, batch_size=32):
    result_name = file_path.split('\\')[-1]
    result_name = result_name[:result_name.rfind('.')] + '_result.jpg'
    result_path = file_path.replace(
        file_path.split('\\')[-1], 'results')
    result_path = os.path.join(result_path, result_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # read an image
    # img = np.asarray(Image.open(file_path))
    im = Image.open(file_path)
    im = im.convert('YCbCr')
    Y = np.array(im)[:, :, 0]

    # read quantization table of Y channel from jpeg images
    qvector = read_q_table(file_path).flatten()

    # load pre-trained weights
    net = Djpegnet(device)
    net.load_state_dict(torch.load(
        './model/djpegnet.pth', map_location=device))
    net.to(device)

    # localizaing using trained detecting double JPEG network.
    result = localizing_double_JPEG(
        Y, qvector, stride=stride, batch_size=batch_size, net=net, device=device)
    print(result.shape)

    # Save the result
    result = result*255
    result = result.astype('uint8')
    img_result = Image.fromarray(result)
    img_result = img_result.resize((Y.shape[1], Y.shape[0]), Image.NEAREST)
    img_result.convert("L")
    img_result.save(result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target', type=str, default='splicing.jpg')
    args = parser.parse_args()

    dir_name = './images'
    file_name = args.target
    result_name = file_name.split('.')[0] + '_result.jpg'
    file_path = os.path.join(dir_name, file_name)
    result_path = os.path.join(dir_name, 'results', result_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # read an image
    # img = np.asarray(Image.open(file_path))
    im = Image.open(file_path)
    im = im.convert('YCbCr')
    Y = np.array(im)[:, :, 0]

    # read quantization table of Y channel from jpeg images
    qvector = read_q_table(file_path).flatten()

    # load pre-trained weights
    net = Djpegnet(device)
    net.load_state_dict(torch.load(
        './model/djpegnet.pth', map_location=device))
    net.to(device)

    # localizaing using trained detecting double JPEG network.
    result = localizing_double_JPEG(
        Y, qvector, stride=args.stride, batch_size=args.batch_size, net=net, device=device)
    print(result.shape)

    # Save the result
    result = result*255
    result = result.astype('uint8')
    img_result = Image.fromarray(result)
    img_result = img_result.resize((Y.shape[1], Y.shape[0]), Image.NEAREST)
    img_result.convert("L")
    img_result.save(result_path)

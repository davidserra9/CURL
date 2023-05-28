import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
from functools import reduce

import numpy as np

import model
from util import ImageProcessing

CHECKPOINT_PATH = "./pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt"
IMAGE_BASE_PATH = "color_naming_images/0000"
OUTPUT_BASE_PATH = "output_images/"

def analogic_or(masks):
    return reduce(lambda x, y: x | y, masks)

def evaluate(img, convert_uint = False):
    """
    Evaluate the modle per image instance. Image of Batch size 1.
    """
    # Load image and convert to tensor
    if isinstance(img, str):
        img = Image.open(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

    img = TF.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        img = img.unsqueeze(0)
        img = torch.clamp(img, 0, 1)

        net_output_img_example, _ = net(img)

        net_output_img_example_numpy = net_output_img_example.squeeze(0).data.cpu().numpy()
        net_output_img_example_numpy = ImageProcessing.swapimdims_3HW_HW3(net_output_img_example_numpy)
        return (net_output_img_example_numpy * 255).astype('uint8') if convert_uint else net_output_img_example_numpy

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build the model
    net = model.CURLNet()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    if DEVICE == 'cuda':
        net = net.cuda()

    # im = cv2.imread("/home/david/Desktop/campclarExt-Night-ajhfgsdkjc8s7atycg23bc2378cgdbsa8.jpg")
    # plt.imshow(im)
    # plt.show()

    COLORS = ['Black', 'Blue', 'Brown', 'Grey', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    COLOR_CAT = [[2, 5, 10], [0, 3, 9], [6, 7], [8], [4], [1]]

    # Load the color naming matrix
    mat = loadmat('/home/david/Downloads/wetransfer_imatges_2023-05-12_1337/Color_naming/ColorNaming/w2c.mat')['w2c']

    im = Image.open('/home/david/Downloads/wetransfer_imatges_2023-05-12_1337/reduced/001.png')
    im = np.array(im)

    index_im = 1+np.floor(im[...,0].flatten()/8).astype(int) + 32*np.floor(im[...,1].flatten()/8).astype(int) + 32*32*np.floor(im[...,2].flatten()/8).astype(int)

    out_images = []
    threshold = 0.1

    for w2cM in mat.T:
        out = w2cM[index_im].reshape(im.shape[:2])

        # out = np.greater_equal(out, threshold).astype(int)

        # output_image = im * np.expand_dims(out, axis=2)
        out_images.append(out)

    color_masks = []
    color_probs = []
    for category in COLOR_CAT:
        color_masks.append(analogic_or(
            [im * np.expand_dims(np.greater_equal(out_images[i], threshold).astype(int), axis=2) for i in
             category]).astype(int))
        color_probs.append(np.sum([out_images[i] for i in category], axis=0))

    curl_images = []
    for color_mask in color_masks:
        curl_images.append(evaluate(Image.fromarray(color_mask.astype(np.uint8)).convert("RGB"), convert_uint=True))

    reconstructed_im = np.sum([np.round((m * np.expand_dims(p, axis=2))).astype('uint8') for m, p in zip(curl_images, color_probs)], axis=0)

    fig = plt.figure(figsize=(20,5))
    plt.subplot(1, 8, 1)
    plt.imshow(im)
    plt.title("Original")

    for idx, (c, t) in enumerate(zip(curl_images, ["Ora-Bro-Yel", "Ach", "Pin-Pur", "Red", "Gre", "Blu"])):
        plt.subplot(1, 8, idx+2)
        plt.title(t)
        plt.imshow(c)

    plt.subplot(1, 8, 8)
    plt.title("Blend w/ probs")
    plt.imshow(reconstructed_im)
    plt.show()



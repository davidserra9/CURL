import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from glob import glob

import matplotlib.pyplot as plt

import numpy as np

import model
from util import ImageProcessing

CHECKPOINT_PATH = "./pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt"
IMAGE_BASE_PATH = "color_naming_images/0000"
OUTPUT_BASE_PATH = "output_images/"
def evaluate(img, convert_uint = False):
    """
    Evaluate the modle per image instance. Image of Batch size 1.
    """
    # Load image and convert to tensor
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

    fig, ax = plt.subplots(len(glob(IMAGE_BASE_PATH + '*ng')), 2)
    for idx, img_path in enumerate(glob(IMAGE_BASE_PATH + '*ng')):
        ax[idx, 0].imshow(np.array(Image.open(img_path)))
        ax[idx, 0].set_title(img_path.split('/')[-1])
        ax[idx, 1].imshow(evaluate(img_path, convert_uint=False))
        ax[idx, 1].set_title('result')
        out_path = OUTPUT_BASE_PATH + img_path.split('/')[-1]
        if out_path[-3:] == 'dng':
            out_path = out_path[:-3] + 'png'

        cv2.imwrite(out_path, evaluate(img_path, convert_uint=True)[..., ::-1])

    plt.show()





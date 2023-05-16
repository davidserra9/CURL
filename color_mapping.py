import torch
from PIL import Image
import torchvision.transforms.functional as TF
from glob import glob

import matplotlib.pyplot as plt

import numpy as np

import model
from util import ImageProcessing

CHECKPOINT_PATH = "./pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt"
IMAGE_BASE_PATH = "/home/david/Downloads/wetransfer_imatges_2023-05-12_1337/Color_naming/ColorNaming/output_images/0000"
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

    im = Image.open("/home/david/Desktop/cerinathus2.png")
    plt.imshow(im)
    plt.show()

    # for idx, img_path in enumerate(glob(IMAGE_BASE_PATH + '*.png')):
    #     print(img_path)
    #     result = evaluate(img_path, convert_uint=False)
    #     plt.imshow(result)
    #     plt.title(img_path.split('/')[-1])
    #     plt.show()




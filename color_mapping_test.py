from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# COLOR ORDER: ['Black', 'Blue', 'Brown', 'Grey', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
# Outputs are 6 images corresponding the the 6 color categories:
# 1. Orange (2) + Brown (5) + Yellow (10)
# 2. Black (0) + Grey (3) + White (9)
# 3. Pink (6) + Purple (7)
# 4. Red(8)
# 5. Green (4)
# 6. Blue (1)

mat = loadmat('/home/david/Downloads/wetransfer_imatges_2023-05-12_1337/Color_naming/ColorNaming/w2c.mat')['w2c']

import matplotlib.pyplot as plt
from matplotlib.image import imsave

import numpy as np

image = np.zeros((100, 100), dtype=bool)
image[25:50, 25:50] = True


imsave("cartographie.png", image)



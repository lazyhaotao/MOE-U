from scipy import io
import numpy as np
from PIL import Image
# load data
imgPth = "../Datasets/KSC/KSC.mat" # Path of Hyperspectral Data
# key: paviaU, pavia, [salinas_corrected, salinas_gt], [salinasA_corrected,salinasA_gt]
# It is difficult to find the rgb channels of other data sets, so just set them casually.
img = io.loadmat(imgPth)['KSC'][:,:,[43,21,11]]
img = np.asarray(img)
print(img.shape)

# Convert hyperspectral data range to rgb data range
img[:,:,0] = img[:,:,0]/np.max(img[:,:,0])*255
img[:,:,1] = img[:,:,1]/np.max(img[:,:,1])*255
img[:,:,2] = img[:,:,2]/np.max(img[:,:,2])*255
img = np.ceil(img)
# convert to PIL image
img = Image.fromarray(np.uint8(img))
img.save("./KSC.png")


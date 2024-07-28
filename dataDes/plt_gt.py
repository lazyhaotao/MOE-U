from scipy import io
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
import os

# If you want to draw the ground truth image, use this code and comment out the following code
# gtPth = '../Datasets/KSC/KSC_gt.mat'  # your data path
# gt = io.loadmat(gtPth)['KSC_gt']
# gt = np.asarray(gt)


# If you want to draw the predicted label image, use this code and comment out the above code
gtPth = " " # your predicted label path
gt = np.load(gtPth)
# set colors of labels
color_list = [(255,255,255),(255,0,0),(107, 142, 35),(155,48,255),(0,0,205),
              (255,165,0),(0,255,0),(0,139,139),(255, 228, 225),
              (139,0,0),(230,230,250),(155,205,155),(25,25,112),(255,193,37)]
color_hex_list = ["#FFFFFF","#FF0000","#6B8E23","#9B30FF","#0000CD",
                  "#FFA500","#00FF00","#008B8B","#FFE4E1",
                  "#8B0000","#E6E6FA","#9BCD9B","#191970","#FFC125"]
# label names
KSC_gt_list= ["undefined","Scrub","Willow-swamp","CP-hammock", "Slash-pine","Oak-Broadleaf","Hardwood","Swap","Graminoid-marsh","Spartina-marsh", "Cattail-marsh","Salt-marsh","Mud-flats","Water"]

# Paint a rgb image based on each number in the gt
gt_rgb = np.zeros((gt.shape[0], gt.shape[1],3), dtype=np.uint8)
for x in range(gt.shape[0]):
    for y in range(gt.shape[1]):
        gt_rgb[x,y,:] = color_list[gt[x,y]]

# number of classes in total: {0,1,2,3,4,5,6,7,8,9}
gt_classes = set(gt.flatten().tolist())
# manually create a legend
patch_list = []
for i in range(1,len(gt_classes)):
    patch_list.append(mpatches.Patch(color=color_hex_list[i], label=KSC_gt_list[i]))

# set the position of the legend
#plt.legend(handles=patch_list, bbox_to_anchor=(1.815, 1.015))
#ncol Set the number of columns to display
fig = plt.figure(figsize=(10, 10))
# Show the legend
#plt.legend(handles=patch_list, bbox_to_anchor=(1.0, 1.2), ncol=5)
gt_rgb = Image.fromarray(gt_rgb)
# do not display axis
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
# save and show the ground truth map
#plt.tight_layout()
plt.imshow(gt_rgb)
#plt.legend()
save_pth = " "
if not os.path.isdir(save_pth):
    os.makedirs(save_pth)
plt.savefig(save_pth+ " picture_name.png")

plt.show()


import numpy as np
import os


def get_sizes_from_one_mask(mask):
    coords = np.argwhere(mask)

    if coords.size == 0:
        h = 0
        w = 0
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        h = y_max - y_min + 1
        w = x_max - x_min + 1

    return int(h), int(w)


def get_sizes_from_one_maskfile(maskfile):
    s=[]
    for mask in maskfile:
        s.extend(get_sizes_from_one_mask(mask))
    return s


def get_sizes_from_all_maskfiles():
    sizes = []
    base_path = './dl_challenge'
    for entry in os.scandir(base_path):
        if entry.is_dir():
            mask_path = os.path.join(entry,'mask.npy')
            maskfile = np.load(mask_path)
            sizes.extend(get_sizes_from_one_maskfile(maskfile))        
    return sizes


sizes = get_sizes_from_all_maskfiles()
print(max(sizes))




# pc = np.load('dl_challenge_example2/pc.npy')
# pc = pc.transpose(1,2,0)
# print(pc.shape)

# mask = np.load('dl_challenge_example2/mask.npy')
# print(mask.shape)

# bbox3d = np.load('dl_challenge_example2/bbox3d.npy')
# print(bbox3d.shape)

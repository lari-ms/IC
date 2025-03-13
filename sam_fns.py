from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sam_fns

def sam_segmentation_function(image, mask_generator):
    '''takes an image, generates and returns a segmentation mask based on the img'''
    image = image.astype('uint8')
    masks = mask_generator.generate(image)
    common_mask = get_common_mask(masks)
    return common_mask

def get_common_mask(anns):
    '''
    takes a list of masks and combines it into a single annotated one (matrix)
    '''
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mm = sorted_anns[0]['segmentation']
    final_msk = np.zeros((mm.shape[0], mm.shape[1]))
    for j, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        final_msk += np.where(m == True, j+1 ,0)
    return final_msk.astype(int)

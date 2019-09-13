import torch
import numpy as np

def alpha_blend(input_image, segmentation_mask, alpha=0.6):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended

def test_time_augmentation(model, image):
    '''
    takes in a single image (1xCxHxW) and flips it horizontally, then 
    returns the combined model outputs of the two images.
    '''
    # flipped image:
    image_flipped = torch.flip(image,dims=[3])

    input = torch.cat((image, image_flipped), dim=0) # concatenate along batch dim 
    outputs = model(input)
   
    outputs_regular = outputs[0]
    # prediction of flipped image: 
    outputs_flipped = outputs[1]
    # flip back (left-right): 
    outputs_flipped_flipped_back = torch.flip(outputs_flipped, dims=[2])
    outputs_combined = (outputs_regular + outputs_flipped_flipped_back) / 2

    return outputs_combined

def test(model,image):
    '''
    simple test :)
    '''
    outputs = model(image)
    return outputs
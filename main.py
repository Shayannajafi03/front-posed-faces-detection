import numpy as np
import matplotlib.pyplot as plt
import cv2 


def Convolution(kernel , image):
    pass


# convolution functoin part
def conv2d(image , kernel):

    """
    Perform a 2D convolution on a 2D image using a given kernel.

    Parameters:
    image (numpy.ndarray): 2D array representing the grayscale image.
    kernel (numpy.ndarray): 2D array representing the kernel.

    Returns:
    numpy.ndarray: 2D array representing the convolved image.
    """

    # flip the kenel horizontaly and vetically --> beacuse differnce in real x , y and input x , y 
    kernel = np.fliplr(np.fliplr(kernel))

    # Get the dimensions of the image and the kernel
    image_height, image_width = image.shape[0] , image.shape[1]
    kernel_height, kernel_width = kernel.shape[0] , kernel.shape[1]

    # Calculate the dimensions of the output
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    #Initialize the output with zeros
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution operation
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest from the image
            region = image[i:i + kernel_height, j:j + kernel_width]
            # Element-wise multiplication and summation
            output[i, j] = np.sum(region * kernel)
    
    return output


def conv3d(image , kernel):
    
    """
    Perform a 2D convolution on a 3D image (e.g., color image) using a given kernel for each channel.

    Parameters:
    image (numpy.ndarray): 3D array representing the image (height x width x channels).
    kernel (numpy.ndarray): 2D array representing the kernel.

    Returns:
    numpy.ndarray: 2D array representing the convolved image.
    """
    # Get the dimensions of the image and the kernel
    image_height, image_width = image.shape[0] , image.shape[1]
    kernel_height, kernel_width = kernel.shape[0] , kernel.shape[1]
    
    # Calculate the dimensions of the output
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize the output with zeros
    output = np.zeros((output_height, output_width , 3))

    # Get the number of channels
    channels = image.shape[2]

    for c in range(channels):
        # Perform convolution for each channel
        convolved_channel = conv2d(image[:, :, c], kernel)
        output[:, :, c] = convolved_channel
        
    min_val = output.min()
    max_val = output.max()
    out_put = (output-min_val)/(max_val - min_val) *255
    
    return out_put.astype(np.int16)

class main():
    def __init__(self):
        pass

    def removing_black_and_white_images(self):
        pass

    def face_detection(self):
        pass

    def face_feature_extraction(self):
        pass

    def is_frontpose(self):
        pass

    def frontposed_faces_detector(self):
        pass
        



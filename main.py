from retinaface import RetinaFace
import matplotlib.pyplot as plt
import numpy as np
import imghdr
import cv2 
import os


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
    
    if output.max() > 255: 
        min_val = output.min()
        max_val = output.max()
        output = (output-min_val)/(max_val - min_val) *255
        
    return output.astype(np.int16)



def is_black_and_white(image):
    # Define the upper and lower bounds for the HSV values
    upper_bound = np.array([259, 30, 100])
    lower_bound = np.array([0, 0, 0])
    
    # Convert the image from BGR to HSV format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask using the inRange function to filter out the specified HSV range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Check the mean value of the mask
    if mask.mean() > 28:
        return True
    return False



def removing_black_and_white_images(path):
        """
        Remove all black and white images from the specified directory.
        """
        for file in os.listdir(path):
              file_path = os.path.join(path, file)
              if imghdr.what(file_path)is not None: # check if it's a image or not
                    image = cv2.imread(file_path)
                    if is_black_and_white(image):
                          os.remove(file_path)
                          print(f"Removed {file_path} because it's black and white")
                          continue
                    



def face_detection(image_path):
        faces = RetinaFace.detect_faces(image_path)
        return faces


def show_faces(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    faces = face_detection(image_path)
    for face in faces:
        face = faces[face]
        x, y, w, h = face['facial_area']
        cv2.rectangle(image, (x, y), (w,h) , (0,255,0) , 2)
    plt.axis("off")
    plt.imshow(image)


def face_feature_extraction():
        pass

def is_frontpose():
        pass

def frontposed_faces_detector():
        pass
        



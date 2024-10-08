from retinaface import RetinaFace
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time
from tqdm import tqdm
import numpy as np
import imghdr
import cv2 
import os


identity_kernel = np.array([
    [0 , 0 , 0],
    [0 , 1 , 0],
    [0 , 0 , 0]
])

leftsobel_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

gaussian_blur_kernel = (1/16)*np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

random_kernel = np.array([
    [0.7 , -1.2 , 0.3],
    [1.1 , -0.5 , 0.8],
    [-0.9 , 0.4 , -0.6]
])

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



def remove_black_and_white_images(path):
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
        x1, y1, x2, y2= face['facial_area']
        cv2.rectangle(image, (x1, y1), (x2,y2) , (0,255,0) , 2)
    plt.axis("off")
    plt.imshow(image)


def face_feature_extraction(face_feature__dic):
    face_features = {}
    face_features["location"] = face_feature__dic["facial_area"]
    face_features["Right_eye"] = face_feature__dic["landmarks"]["right_eye"]
    face_features["Left_eye"] = face_feature__dic["landmarks"]["left_eye"]
    face_features["Nose"] = face_feature__dic["landmarks"]["nose"]
    face_features["mouth_right"] = face_feature__dic["landmarks"]["mouth_right"]     
    face_features["mouth_left"] = face_feature__dic["landmarks"]["mouth_left"]
    
    return face_features



def is_frontpose(useful_features):
        
    """
    Determines if a face is in a frontal pose based on the positions of key landmarks.

    Steps:
    1. Calculate the Euclidean distance between the left and right eyes (d_eyes).
    2. Calculate the midpoint between the eyes (mid_eyes_x).
    3. Define a threshold for frontal face alignment as a small fraction (e.g., 35%) of the inter-eye distance (d_eyes).
    4. Calculate the horizontal distances between the nose , mouse and the midpoint of the eyes (d_nose_mid_eyes , d_nose_mid_eyes).
    6. Compare the distance d_nose_mid_eyes  with the defined threshold.
    7. If its distance is less than the threshold, the face is considered to be in a frontal pose.
    """
    Right_eye = np.array(useful_features["Right_eye"])
    Left_eye = np.array(useful_features["Left_eye"])
    Nose = np.array(useful_features["Nose"])
    mouth_right = np.array(useful_features["mouth_right"])
    mouth_left = np.array(useful_features["mouth_left"])


    d_eyes = norm(Right_eye-Left_eye)
    mid_eyes_x = (Right_eye[0] + Left_eye[0]) / 2
    mid_mouse_x = (mouth_right[0] + mouth_left[0]) / 2

    # Define threshold as 35% of the inter-eye distance
    threshold = 0.35 * d_eyes

    # Calculate horizontal distance from the nose to the midpoint of the eyes
    d_nose_mid_eyes = abs(Nose[0] - mid_eyes_x)
    d_mouse_mid_eyes = abs(mid_mouse_x - mid_eyes_x)

    if (d_nose_mid_eyes <= threshold) and (d_mouse_mid_eyes <= threshold):
         return True
    
    return False



def save_croped_image(file , croped_image):
    # Save the cropped image to a file
    SAVE_PATH = "./front pose images"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    #generate unique name for each face
    image_name = f"{file.split('.')[0]} , image_{int(time.time() * 1000)}.jpg"
    image_path = os.path.join(SAVE_PATH , image_name)
    plt.imsave(image_path , croped_image)




def frontposed_faces_detector(folder_path):
    count = 0 # Initialize a counter for the number of front pose faces found
    for file in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)

        if imghdr.what(file_path)is not None: # check if it's a image or not
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            faces_features = face_detection(file_path) # collect all faces features

            for face_feature in faces_features:
                face_info = faces_features[face_feature] # collect features of each face
                useful_features = face_feature_extraction(face_info) # collect useful features of each face

                if is_frontpose(useful_features):
                    x1 , y1 , x2 , y2 = useful_features["location"] 
                    croped_image = image[y1:y2 , x1:x2]
                    count += 1
                    save_croped_image(file , croped_image)
                    
                        
                        
    print(f"find {count} front pose face(s)")



if __name__ == "__main__":
    #to show how convolution function works
    test_img = cv2.cvtColor(cv2.imread("./faces/3.webp") , cv2.COLOR_BGR2RGB)
    kernel_list = [
        identity_kernel,
        leftsobel_kernel,
        gaussian_blur_kernel,
        random_kernel
    ]

    kernel_names = ["identity_kernel" , "leftsobel_kernel" , "gaussian_blur_kernel" , "random_kernel"]
    plt.figure(figsize=(10 , 5))
    for idx in range(4):
        ax = plt.subplot(2 , 2 , idx+1)
        img = conv3d(test_img , kernel_list[idx])
        ax.set_title(kernel_names[idx])
        ax.imshow(img)
        ax.axis("off")

    plt.show()
        
    folder_path = "./faces"
    remove_black_and_white_images(folder_path)
    frontposed_faces_detector(folder_path)




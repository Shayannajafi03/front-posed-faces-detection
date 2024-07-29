# Face Detection and Pose Estimation

This project focuses on face detection, removal of black and white images, and determination of frontal face poses from a given set of images. The primary functions include convolution operations, face detection, face feature extraction, and frontal pose determination.

## Features

1. **2D Convolution on Grayscale Images**
2. **2D Convolution on Color Images**
3. **Black and White Image Detection and Removal**
4. **Face Detection using RetinaFace**
5. **Face Feature Extraction**
6. **Frontal Pose Detection**
7. **Cropping and Saving Frontal Faces**

## Installation

To get started with the project, you need to install the required dependencies. Use the following command to install them:

```bash
pip install retinaface matplotlib numpy tqdm opencv-python
```

## Usage

### 1. Convolution Functions

- **2D Convolution (Grayscale)**

  ```python
  from main import conv2d
  convolved_image = conv2d(image, kernel)
  ```

- **3D Convolution (Color)**

  ```python
  from main import conv3d
  convolved_image = conv3d(image, kernel)
  ```

### 2. Removing Black and White Images

To remove black and white images from a directory, use:

```python
from main import remove_black_and_white_images
remove_black_and_white_images(path)
```

### 3. Face Detection and Frontal Pose Detection

To detect faces and determine if they are in a frontal pose, use:

```python
from main import frontposed_faces_detector
frontposed_faces_detector(folder_path)
```

### Example

An example script to demonstrate the workflow:

```python
if __name__ == "__main__":
    folder_path = "./faces"
    remove_black_and_white_images(folder_path)
    frontposed_faces_detector(folder_path)
```

### Functions

#### `conv2d(image, kernel)`

- **Parameters:**
  - `image`: 2D numpy array representing the grayscale image.
  - `kernel`: 2D numpy array representing the kernel.
- **Returns:** 2D numpy array representing the convolved image.

#### `conv3d(image, kernel)`

- **Parameters:**
  - `image`: 3D numpy array representing the image (height x width x channels).
  - `kernel`: 2D numpy array representing the kernel.
- **Returns:** 2D numpy array representing the convolved image.

#### `remove_black_and_white_images(path)`

- **Parameters:**
  - `path`: Directory path containing images.
- **Description:** Removes all black and white images from the specified directory.

#### `face_detection(image_path)`

- **Parameters:**
  - `image_path`: Path to the image file.
- **Returns:** Dictionary containing detected faces and their features.

#### `frontposed_faces_detector(folder_path)`

- **Parameters:**
  - `folder_path`: Directory path containing images.
- **Description:** Detects faces and determines if they are in a frontal pose, then crops and saves them.

## Directory Structure

```
.
├── main.py
├── README.md
├── faces/                # Directory containing the input images
└── front pose images/    # Directory containing cropped frontal face images
```

## Contributing

If you would like to contribute to this project, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries or questions, please contact Shayan najafi at  najfishayan492@gmail.com.

---

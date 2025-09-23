import cv2
import numpy as np

def preprocess_image(image_path, input_shape=(1, 3, 640, 640)):
    """
    Loads and preprocesses an image:
    - Resize to target width, height
    - Convert BGR to RGB
    - Normalize pixel values (0-1)
    - Convert to CHW layout
    - Add batch dimension
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found")

    # Resize image (width, height)
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0,1]
    img = img.astype(np.float32) / 255.0

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

def save_tensor_to_bin(tensor, filename):
    """
    Saves the tensor to a binary file in float32 raw format
    """
    tensor.tofile(filename)
    print(f"Saved tensor to {filename} with shape {tensor.shape} and dtype {tensor.dtype}")

if __name__ == '__main__':
    image_path = 'input.jpg'              # Path to your input image
    output_bin_path = 'input.bin'         # Output binary file for trtexec input

    # Preprocess image to tensor
    tensor = preprocess_image(image_path)

    # Save to binary file
    save_tensor_to_bin(tensor, output_bin_path)


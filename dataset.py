import cv2, os
from config import IMG_WIDTH, IMG_HEIGHT

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    # Path to data folder
    data_path = os.path.join(data_dir)

    # Number of subdirectories/labels
    number_of_labels = 0
    
    for i in os.listdir(data_path):
        number_of_labels += 1

    # Loop through the subdirectories
    for sub in range(number_of_labels):
        sub_folder = os.path.join(data_path, str(sub))

        images_in_subfolder = []

        for image in os.listdir(sub_folder):
            images_in_subfolder.append(image)

        # Open each image 
        for image in images_in_subfolder:
            image_path = os.path.join(sub_folder, image)
            
            #Doc anh bang OpenCV
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # Add Label
            labels.append(sub)
            
            # Resize and Add Image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            
            # print(image_path)
            
    return (images, labels)
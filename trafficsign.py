import sys
import numpy as np
import tensorflow as tf
import cv2
import os
from utils import predict_real_image

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Nếu truyền vào file .h5 có sẵn → load luôn, không train
    if len(sys.argv) == 3 and sys.argv[2].endswith(".h5") and os.path.exists(sys.argv[2]):
        print(f"Tim thay model: {sys.argv[2]} → Bo qua train, load model...")
        model = tf.keras.models.load_model(sys.argv[2])
        predict_real_image(model)
        return
    
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    history = model.fit(x_train, y_train,epochs=EPOCHS,validation_data=(x_test, y_test))
      
    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Ve do thi
    plot_history(history)
    
    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

    # Nhan dien anh thuc te 
    predict_real_image(model)

if __name__ == "__main__":
    main()

import sys, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split

from config  import TEST_SIZE, EPOCHS
from dataset import load_data
from model   import get_model
from utils   import plot_history, predict_real_image

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

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

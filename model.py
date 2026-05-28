import tensorflow as tf
from config import IMG_WIDTH, IMG_HEIGHT, NUM_CATEGORIES

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([

        # Convolutional layers and Max-pooling layers
        tf.keras.layers.Conv2D(32, (3,3), activation="relu",input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Hidden Layers
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        
        # Dropout
        tf.keras.layers.Dropout(0.5),
              
        # Output layer with output units for all digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def plot_history(history):
    """Ve do thi Accuracy va Loss theo epoch"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Traffic Sign Recognition - Training Results",
                 fontsize=14, fontweight="bold")
    epochs_range = range(1, EPOCHS + 1)

    ax1.plot(epochs_range, history.history["accuracy"],
             "b-o", label="Train Accuracy", linewidth=2)
    ax1.plot(epochs_range, history.history["val_accuracy"],
             "r-o", label="Val Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    ax2.plot(epochs_range, history.history["loss"],
             "b-o", label="Train Loss", linewidth=2)
    ax2.plot(epochs_range, history.history["val_loss"],
             "r-o", label="Val Loss", linewidth=2)
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150, bbox_inches="tight")
    print("Da luu do thi: training_results.png")
    plt.show()

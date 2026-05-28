import os          # ← thêm dòng này
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import EPOCHS, IMG_WIDTH, IMG_HEIGHT, SIGN_NAMES

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


def predict_real_image(model):
    """Nhan dien anh bien bao tu ben ngoai dataset"""
    print("\n" + "="*50)
    print("  NHAN DIEN ANH THUC TE")
    print("="*50)

    while True:
        img_path = input("\nNhap duong dan anh ('q' de thoat): ").strip()
        if img_path.lower() == "q":
            break
        if not os.path.exists(img_path):
            print("Khong tim thay file!")
            continue

        img_original = cv2.imread(img_path)
        if img_original is None:
            print("Khong doc duoc anh!")
            continue

        img_resized = cv2.resize(img_original, (IMG_WIDTH, IMG_HEIGHT))
        img_input = np.expand_dims(img_resized, axis=0)

        predictions = model.predict(img_input, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class]) * 100
        sign_name = SIGN_NAMES.get(predicted_class, f"Khong xac dinh ({predicted_class})")

        print(f"\n  Ket qua nhan dien:")
        print(f"  Loai bien bao : [{predicted_class:2d}] {sign_name}")
        print(f"  Do tin cay    : {confidence:.2f}%")

        # Top 3
        top3 = np.argsort(predictions[0])[::-1][:3]
        print("\n  Top 3 du doan:")
        for i, idx in enumerate(top3, 1):
            name = SIGN_NAMES.get(int(idx), f"Class {idx}")
            print(f"    {i}. [{idx:2d}] {name}: {predictions[0][idx]*100:.1f}%")

        # Hien thi anh
        display = cv2.resize(img_original, (500, 430))
        cv2.rectangle(display, (0, 0), (500, 85), (0, 0, 0), -1)
        cv2.putText(display, f"[{predicted_class}] {sign_name[:35]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        cv2.putText(display, f"Do tin cay: {confidence:.1f}%", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("Ket qua nhan dien - nhan phim bat ky de dong", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
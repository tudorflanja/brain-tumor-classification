# Brain Tumor Classification (MRI) using CNN

This project implements a deep learning pipeline for classifying brain tumors from MRI images using a Convolutional Neural Network (CNN). The dataset includes four categories: `glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`, and `no_tumor`.

## 📁 Dataset

The dataset was preprocessed and structured using a custom PyTorch Dataset class with lazy loading. Images were resized, normalized, and converted to tensors to ensure compatibility with the CNN model.

- Training samples: 2870
- Testing samples: 394
- Validation split: 80% train / 20% validation

## ⚙️ Preprocessing Steps

- Converted grayscale images to RGB
- Resized all images to 256×256 pixels
- Normalized pixel values to range [0, 1]
- Applied the following transformations:
  - Gaussian Blur (denoise)
  - CLAHE (contrast enhancement)
  - Intensity normalization
  - Sobel edge detection
  - Brightness adjustment

## 📊 Class Distribution

Each data subset (train, validation, test) was analyzed for class balance. While mostly balanced, minor variations were noted. Strategies like data augmentation and class weighting are recommended to mitigate imbalance.

## 🧠 CNN Model

A custom CNN was trained on the dataset using:
- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Epochs: 10
- Batch size: 32

### Evaluation Metrics (on test set):
- Accuracy: **84%**
- Best validation accuracy: **85.04%**
- Additional metrics: Precision, Recall, F1-score, Confusion Matrix

## 🖼️ Visualizations

Included visual examples:
- Input image samples per class
- Training/validation loss curves
- Accuracy evolution
- Confusion matrix and classification report

## 📝 Conclusion

The model demonstrated strong performance across all tumor categories. Further improvements can be achieved through advanced data augmentation and hyperparameter tuning.

## 📂 Files in This Repository

- `main.py` – contains the complete training and evaluation pipeline
- `Raport.pdf` – full project documentation including figures and analysis

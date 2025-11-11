# 🧬 Brain Tumor Classification (MRI) using CNN

This project implements a deep learning pipeline for classifying brain tumors from MRI images using a Convolutional Neural Network (CNN). The dataset includes four categories: `glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`, and `no_tumor`.

---

## 📁 Dataset

The dataset was preprocessed and structured using a custom PyTorch Dataset class with lazy loading. Images were resized, normalized, and converted to tensors to ensure compatibility with the CNN model.

- Training samples: 2870
- Testing samples: 394
- Validation split: 80% train / 20% validation

---

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

---

## 📊 Class Distribution

Each data subset (train, validation, test) was analyzed for class balance. While mostly balanced, minor variations were noted. Strategies like data augmentation and class weighting are recommended to mitigate imbalance.

---

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

---

- ## 📂 Files in This Repository

- `README.md` – includes an overview of the project, setup instructions, and usage guide
- `Raport.pdf` – full project documentation including figures and analysis
- `main.py` – contains the complete training and evaluation pipeline

---

## 🖼️ Visualizations

Included visual examples:
- Input image samples per class
- Training/validation loss curves
- Accuracy evolution
- Confusion matrix and classification report

---

## ▶️ How to Run the Project in Google Colab

You can run this project interactively using Google Colab by following these steps:

1. **Open the Notebook in Colab**  
   If you have a version of the code converted into a notebook (`.ipynb`), upload it to your GitHub repository or Google Drive. Then open it in Colab using:  
   - GitHub:  
     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/your-notebook-name.ipynb)

   - Google Drive:  
     Go to [https://colab.research.google.com](https://colab.research.google.com), select the **"Upload"** or **"Google Drive"** tab, and open your `.ipynb` file.

2. **Install Required Libraries**  
   In the first code cell, install any missing dependencies:
   ```python
   !pip install torch torchvision matplotlib scikit-learn opencv-python
   ```

3. **Upload or Mount Dataset**  
   If the dataset is local:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
   Or mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Run the Training Pipeline**  
   Execute each cell sequentially, or use `Runtime > Run all` to start training and evaluation.

5. **Visualize Results**  
   Plots for loss, accuracy, and the confusion matrix will be displayed inline in the notebook.

> 💡 Google Colab provides free access to GPUs via `Runtime > Change runtime type > GPU`, which can accelerate training significantly.

---

## 📝 Conclusion

The model demonstrated strong performance across all tumor categories. Further improvements can be achieved through advanced data augmentation and hyperparameter tuning.

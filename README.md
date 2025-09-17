# 🖼️ Image Classification with CNNs

## 📌 Project Overview
This project demonstrates the use of **Convolutional Neural Networks (CNNs)** for image classification.  
It covers three main tasks:

1. **MNIST Digit Classification** – Handwritten digit recognition (0–9).  
2. **CIFAR-10 Object Classification** – Classifying 10 real-world object categories.  
3. **Custom Image Prediction** – Testing the trained model with external images.  

---

## 🚀 Technologies Used
- Python 3  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- scikit-learn  

---

## 📂 Dataset
- **MNIST**: 70,000 grayscale digit images (28x28).  
- **CIFAR-10**: 60,000 color images (32x32, 10 classes).  
- **Custom Images**: Downloaded from the internet and resized for prediction.  

---

## ⚙️ Methodology
1. Data Preprocessing (normalization, reshaping).  
2. CNN Model Building using Conv2D, MaxPooling2D, Flatten, Dense.  
3. Training with Adam optimizer and Sparse Categorical Crossentropy loss.  
4. Evaluation: Accuracy, Loss curves, Confusion Matrix, Classification Report.  
5. Custom Image Prediction from internet or local uploads.  

---

## 📊 Results
### MNIST:
- ✅ Test Accuracy: ~99%  
- Very few misclassifications.  

### CIFAR-10:
- ✅ Test Accuracy: ~70% (simple CNN baseline).  
- Some confusion between similar classes (cat vs. dog, automobile vs. truck).  

### Custom Images:
- ✅ Successfully predicts categories (e.g., dog, car, airplane).  

---

## 🔮 Future Work
- Add **Data Augmentation** (rotation, flipping, zoom).  
- Use **Transfer Learning** with pretrained models (VGG16, ResNet, MobileNet).  
- Deploy model as a **Web App** (Flask, Django, Streamlit).  
- Extend to larger datasets (CIFAR-100, ImageNet).  

---

## 📌 Project Structure
```
├── notebook.ipynb       # Google Colab Notebook with full code
├── README.md            # Project documentation (this file)
└── sample_images/       # Custom test images (optional)
```

---

## ▶️ How to Run
1. Open the notebook in **Google Colab**.  
2. Run cells step by step (MNIST → CIFAR-10 → Custom Prediction).  
3. Upload your own images or use given URLs for testing.  

---

## ✨ Author
👨‍💻 Developed by **SHREYASH PATIL**  
📌 Data Science & Machine Learning Enthusiast  

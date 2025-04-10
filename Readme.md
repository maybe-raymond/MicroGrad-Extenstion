# Microgrid Library Extension

This repository contains an extended version of the Microgrid Library. In this extension, we have implemented additional features to enhance the functionality and usability of the library for machine learning applications. We have also included a detailed comparison notebook to evaluate the performance of the custom neural network implemented in this library against the scikit-learn Multi-Layer Perceptron (MLP) classifier in the `MLP comparison.ipynb file`.

---

## New Features

### 1. **Activation Functions**
   - **ReLU (Rectified Linear Unit):** Added as an activation function.
   - **Sigmoid:** Added for applications requiring probabilistic outputs or binary classification.

### 2. **Weight Initialization**
   - **Xavier Initialization:** Introduced to improve the convergence rate of the neural network by initializing weights effectively.

### 3. **Custom Optimizer**
   - Implemented support for **Lasso Regularization** to handle overfitting.
   - Developed an **SGD with Momentum** optimizer to speed up convergence and escape local minima.

---

## Performance Comparison

We provide a Jupyter Notebook that compares the performance of the following:

1. **Scikit-learn MLP Classifier:**
   - A widely used, well-optimized neural network implementation from scikit-learn.

2. **Custom Neural Network Implementation:**
   - Built using the extended Microgrid Library.
   - Incorporates the newly added ReLU and Sigmoid activation functions, Xavier Initialization, Lasso Regularization, and SGD with Momentum.

### Notebook Features
- Side-by-side comparison of training and validation accuracies.
- Analysis of convergence rates for both models.
- Evaluation metrics such as accuracy, precision, recall, and F1-score.

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Jupyter Notebook
- NumPy
- scikit-learn

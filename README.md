# 🧠 Breast Cancer Classification using Machine Learning and PyTorch

A complete end-to-end pipeline for diagnosing breast cancer using the **Wisconsin Breast Cancer (Original)** dataset. This project consists of **data preprocessing**, implementation of **classical models**, and a detailed **neural network (MLP)** approach with **hyperparameter tuning** and overfitting control, developed in **PyTorch**.

---

## 📌 Table of Contents

- [Part 1: Classical ML Models & Preprocessing](#part-1-classical-ml-models--preprocessing)
- [Part 2: Deep Learning with PyTorch](#part-2-deep-learning-with-pytorch)
- [Dataset Description](#dataset-description)
- [Project Files](#project-files)


---

## ✅ Part 1: Classical ML Models & Preprocessing

Using the [Wisconsin Breast Cancer (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) dataset, the following steps were performed:

### 🔹 Step-by-step Pipeline:

- ✅ Identify and remove missing rows
- ✅ Handle missing values (`?`) and convert to numerical
- ✅ Detect and manage **outliers**
- ✅ Normalize features using **Min-Max Scaling**
- ✅ Split the dataset into:
  - **Training set (80%)**
  - **Test set (20%)**

### 🧠 Classical Algorithms:

- **Perceptron**
- **Multi-Layer Perceptron (MLP)**

Models were trained and validated using **7-fold cross-validation**.

### 📊 Evaluation Metrics:

- Accuracy
- Precision / Recall / F1-score / F2-score
- Confusion Matrix
- **AUC (Area Under ROC Curve)**

---

## 🔬 Part 2: Deep Learning with PyTorch

In the second part, we transferred the preprocessed data to a **PyTorch MLP model** and studied the influence of multiple hyperparameters.

### 🔧 Hyperparameter Grid Search

| Parameter             | Values                             |
|-----------------------|------------------------------------|
| Hidden Layers         | 1, 3                                |
| Hidden Neurons        | Two values per config (e.g., 16, 32) |
| Activation Functions  | Tanh, Leaky ReLU                   |
| Learning Rates        | 0.01, 0.001                        |
| Optimizer             | Adam                               |
| Loss Function         | Binary Cross-Entropy               |

> 🔢 Total Models Tested: **2 × 2 × 2 × 2 = 16** combinations

Each model was evaluated using:
- Confusion Matrix
- AUC
- Accuracy & Recall

### 🛡 Handling Overfitting

In case of overfitting:
- `Dropout` was added
- `L2 Regularization` was applied

Final results reported the **best performing model** based on **test performance and robustness**.

---

## 🗂 Dataset Description

- **Name:** Wisconsin Breast Cancer (Original)  
- **Source:** University of Wisconsin Hospitals, Madison  
- **Donor:** Dr. William H. Wolberg  
- **Instances:** 699  
- **Attributes:** 10 predictive features + 1 target class  
- **Missing Values:** 16 entries had one missing value (denoted as `?`)  
- **Class Distribution:**
  - `2` = Benign (65.5%)
  - `4` = Malignant (34.5%)

### 🔬 Features

1. **Sample Code Number** – Unique ID for the sample (not a predictive feature)  
2. **Clump Thickness**  
3. **Uniformity of Cell Size**  
4. **Uniformity of Cell Shape**  
5. **Marginal Adhesion**  
6. **Single Epithelial Cell Size**  
7. **Bare Nuclei**  
8. **Bland Chromatin**  
9. **Normal Nucleoli**  
10. **Mitoses**  
11. **Class** – Target variable (`2 = Benign`, `4 = Malignant`)

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `BreastCancer_Wisconsin_Pipeline_MLP_Tuning.ipynb` | Main Jupyter notebook |
| `breast_cancer_wisconsin_original.csv` | dataset |
| `breast-cancer-wisconsin.names` | Original UCI dataset description |
| `README.md` | Project documentation |

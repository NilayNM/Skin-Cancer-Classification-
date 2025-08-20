# Skin-Cancer-Classification-

This repository contains the code and results for our group research project:  
**"Enhanced Multiclass Skin Cancer Classification Using EfficientNet-B2 with Advanced Data Handling"**

---

## 📌 Project Overview
Skin cancer is one of the most common cancers worldwide, and early detection is critical for improving patient outcomes.  
We implemented a **deep learning model using EfficientNet-B2** to classify dermoscopic images of skin lesions into **6 classes**:

- Melanoma  
- Nevus  
- Basal Cell Carcinoma (BCC)  
- Actinic Keratosis (AK)  
- Benign Keratosis (BK)  
- Vascular Lesions  

Our framework integrates:
- **Hybrid preprocessing pipeline** with resizing, augmentation, normalization  
- **Weighted Random Sampling** to address class imbalance  
- **Fine-tuning of EfficientNet-B2** with a custom classifier head  
- Robust training with **AdamW optimizer, learning-rate scheduling, and early stopping**  

---

## 📊 Results
- **Overall Accuracy**: 82%  
- **Per-class ROC-AUC**: 0.94 – 0.98  
- **Key Insights**:
  - High AUC for **melanoma (0.96)** and **basal cell carcinoma (0.98)**  
  - Balanced precision/recall across classes, with improvements for minority lesion types using class balancing  

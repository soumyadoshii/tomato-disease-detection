# 🍅 Tomato Leaf Disease Detection with Deep Learning & Explainable AI

An end-to-end deep learning framework for **tomato leaf disease detection, classification, and explainability** across **11 classes** — 10 disease categories and 1 healthy category.

The project benchmarks five state-of-the-art deep learning architectures — **ResNet50, EfficientNet-B0, DenseNet-121, ConvNeXt-Tiny, and Vision Transformer (ViT-B/16)** — using comprehensive multi-class evaluation metrics.

To address the **black-box nature of deep learning models**, the framework integrates **Grad-CAM** and **LIME** to provide visual explanations of model predictions.

---

## 📌 Project Overview

### 🔬 Comparative Architecture Benchmarking

Evaluates five CNN and Transformer-based architectures to compare their classification performance, robustness, and generalization across tomato leaf disease classes.

### 🧠 Explainable AI Integration

Integrates:

* **Grad-CAM** — generates heatmaps highlighting regions that contributed most to the model's prediction.
* **LIME** — uses superpixel-based explanations to identify image regions influencing the classification.

### 🌐 Interactive Inference Interface

Includes a **Streamlit web application** that allows users to:

* Upload tomato leaf images
* Obtain disease predictions
* View prediction confidence
* Generate visual XAI explanations using Grad-CAM and LIME

### 🏗️ Modular Pipeline

The codebase separates major components into dedicated modules for:

* Dataset preprocessing
* Data augmentation
* Model construction
* Training and evaluation
* Explainability
* Web-based inference

---

## 📊 Dataset

The project uses a tomato leaf image dataset containing **32,531 images** distributed across **11 target classes**.

| #  | Disease / Class                        |
| -- | -------------------------------------- |
| 1  | `Bacterial_spot`                       |
| 2  | `Early_blight`                         |
| 3  | `Late_blight`                          |
| 4  | `Leaf_Mold`                            |
| 5  | `Septoria_leaf_spot`                   |
| 6  | `Spider_mites Two-spotted_spider_mite` |
| 7  | `Target_Spot`                          |
| 8  | `Tomato_Yellow_Leaf_Curl_Virus`        |
| 9  | `Tomato_mosaic_virus`                  |
| 10 | `healthy`                              |
| 11 | `powdery_mildew`                       |

The dataset is organized into **training, validation, and test splits** for standardized model development and evaluation.

---

## 🏗️ Model Architectures

Five architectures are benchmarked within a unified PyTorch pipeline:

| Model               | Architecture Type            |
| ------------------- | ---------------------------- |
| **ResNet50**        | Convolutional Neural Network |
| **EfficientNet-B0** | Efficient CNN                |
| **DenseNet-121**    | Dense CNN                    |
| **ConvNeXt-Tiny**   | Modern CNN                   |
| **ViT-B/16**        | Vision Transformer           |

All models are trained using:

* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** AdamW
* **Framework:** PyTorch
* **Task:** 11-class image classification

---

## 📈 Model Performance & Comparative Benchmark

Models are evaluated using standard multi-class classification metrics.

| Model Architecture  |   Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |  ROC-AUC (OvR) |
| ------------------- | -------------: | ----------------: | -------------: | ---------------: | -------------: |
| **ResNet50**        | *Benchmarking* |    *Benchmarking* | *Benchmarking* |   *Benchmarking* | *Benchmarking* |
| **EfficientNet-B0** | *Benchmarking* |    *Benchmarking* | *Benchmarking* |   *Benchmarking* | *Benchmarking* |
| **DenseNet-121**    | *Benchmarking* |    *Benchmarking* | *Benchmarking* |   *Benchmarking* | *Benchmarking* |
| **ConvNeXt-Tiny**   | *Benchmarking* |    *Benchmarking* | *Benchmarking* |   *Benchmarking* | *Benchmarking* |
| **ViT-B/16**        | *Benchmarking* |    *Benchmarking* | *Benchmarking* |   *Benchmarking* | *Benchmarking* |

> **Note:** Performance values will be updated after completion of model benchmarking.

---

## 💡 Explainable AI

A major component of this project is understanding **why** a model makes a particular prediction rather than treating the model as a black box.

### 🔥 Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** generates a spatial heatmap showing the regions of an input image that contributed most strongly to the predicted class.

For tomato leaf disease detection, this helps determine whether the model is focusing on relevant symptomatic regions such as:

* Lesions
* Discoloration
* Spots
* Mold patterns
* Leaf texture abnormalities

### 🧩 LIME

**Local Interpretable Model-agnostic Explanations (LIME)** provides local explanations by segmenting the image into **superpixels** and determining which regions have the greatest influence on the prediction.

Together, Grad-CAM and LIME provide complementary explanations of model behavior.

---

## 📁 Project Structure

```text
tomato-disease-detection/
│
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── saved_models/
│   └── *.pt
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train_eval.py
│   └── explainability.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Module Description

| File / Directory    | Purpose                                          |
| ------------------- | ------------------------------------------------ |
| `data/`             | Training, validation, and test datasets          |
| `saved_models/`     | Stores trained PyTorch model checkpoints         |
| `dataset.py`        | Dataset loading, preprocessing, and augmentation |
| `models.py`         | Model construction for all five architectures    |
| `train_eval.py`     | Training pipeline and evaluation metrics         |
| `explainability.py` | Grad-CAM and LIME implementation                 |
| `app.py`            | Streamlit inference interface                    |
| `requirements.txt`  | Python dependencies                              |
| `README.md`         | Project documentation                            |

---

## ⚡ Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/soumyadoshii/tomato-disease-detection.git
cd tomato-disease-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the dataset inside the `data/` directory using the following structure:

```text
data/
├── train/
│   ├── Bacterial_spot/
│   ├── Early_blight/
│   ├── ...
│   └── healthy/
│
├── valid/
│   ├── Bacterial_spot/
│   ├── Early_blight/
│   ├── ...
│   └── healthy/
│
└── test/
    ├── Bacterial_spot/
    ├── Early_blight/
    ├── ...
    └── healthy/
```

### 4. Verify Dataset Loading

```bash
python src/dataset.py
```

### 5. Train & Evaluate a Model

Run the training and evaluation pipeline:

```bash
python src/train_eval.py
```

The trained model checkpoints are saved in:

```text
saved_models/
```

### 6. Launch the Streamlit Application

Start the interactive inference interface:

```bash
python -m streamlit run app.py
```

The application allows users to upload a tomato leaf image and view:

**Input Image → Disease Prediction → Confidence → Grad-CAM → LIME Explanation**

---

## 🔬 End-to-End Pipeline

```text
                 ┌─────────────────────┐
                 │   Tomato Leaf Image  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Preprocessing &     │
                 │ Data Augmentation   │
                 └──────────┬──────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │   Deep Learning Models       │
             │                              │
             │ ResNet50                     │
             │ EfficientNet-B0              │
             │ DenseNet-121                 │
             │ ConvNeXt-Tiny                │
             │ ViT-B/16                     │
             └──────────────┬───────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Disease Prediction  │
                 │    + Confidence     │
                 └──────────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
           ┌──────────────┐    ┌──────────────┐
           │   Grad-CAM   │    │     LIME     │
           │  Heatmap     │    │  Superpixels │
           └──────┬───────┘    └──────┬───────┘
                  │                   │
                  └─────────┬─────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Explainable Disease │
                 │      Diagnosis      │
                 └─────────────────────┘
```

---

## 🎯 Objectives

The project aims to:

1. **Develop** an automated deep learning system for tomato leaf disease classification.
2. **Benchmark** CNN and Vision Transformer architectures under a standardized evaluation pipeline.
3. **Compare** models using accuracy, precision, recall, F1-score, and ROC-AUC.
4. **Improve model transparency** through Grad-CAM and LIME explanations.
5. **Deploy** the trained models through an interactive Streamlit application.
6. **Investigate model attention** to determine whether predictions are based on meaningful symptomatic regions.

---

## 🚀 Future Scope

Potential extensions include:

* Integration of **IoT-based environmental sensing** such as temperature, humidity, and rainfall.
* Incorporation of **weather and field-level contextual information**.
* Disease **severity estimation** in addition to classification.
* Leaf **segmentation** before disease classification.
* Deployment on **edge devices** for field-level inference.
* Integration of additional crop species and disease categories.
* Development of a lightweight model suitable for **mobile or embedded deployment**.

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **Scikit-learn**
* **OpenCV**
* **LIME**
* **Grad-CAM**
* **Streamlit**
* **NumPy**
* **Pandas**
* **Matplotlib**

---

## 📌 Project Status

**Current Status:** 🚧 Active Development

The model benchmarking and evaluation stage is currently in progress. Final performance metrics will be added once all five architectures have been trained and evaluated under the same experimental setup.

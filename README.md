<img width="1155" height="628" alt="image" src="https://github.com/user-attachments/assets/f3091afa-e311-4394-81a3-a4624cf1a66b" />

This repository outlines the transition from basic Artificial Neural Networks (ANN) to Convolutional Neural Networks (CNNs) and demonstrates the implementation of Transfer Learning  using the EfficientNet architecture to solve the Fashion MNIST classification task.

# 🚀 Fashion-CNN: Advanced Classification with EfficientNetB0

While basic neural networks are functional for simple datasets, images possess **Spatial Hierarchies (Mekansal Hiyerarşiler)** that standard fully connected layers are not well-equipped to capture. Standard networks treat images as flat vectors, losing spatial context, whereas **Convolutional Neural Networks (CNNs)** apply convolutional filters to help the model learn spatial features such as edges, textures, and patterns. 

This project implements a CNN-based approach using **Transfer Learning (Transfer Öğrenme)** to recognize these complex structures in the Fashion MNIST dataset.

---

## 🎯 Key Learning Objectives

* **Understanding CNN Superiority:** Learning why CNNs are more effective for image data compared to standard neural networks.
* **Implementing Transfer Learning:** Leveraging pre-trained weights from massive datasets to solve a targeted task.
* **Utilizing EfficientNet-B0:** Using the lightest version of a state-of-the-art CNN architecture as a high-efficiency feature extractor.
* **Executing Fine-tuning (İnce Ayar):** Adapting a pre-trained model specifically to the fashion classification domain.

---

## 🧠 Core Concepts

### 1. The Power of CNNs
CNNs are specifically designed for image recognition because they recognize patterns hierarchically—building from simple edges to complex shapes. However, training them from scratch is computationally expensive and requires millions of labeled data points.



### 2. Transfer Learning & Fine-Tuning
Transfer Learning is a technique where a model trained on a massive dataset, such as ImageNet, is reused for a new task. Instead of training from scratch, we **Fine-tune (İnce Ayar)** the model on our specific dataset.
* **Efficiency:** Saves significant time and computational resources.
* **Performance:** Often leads to better results, especially on targeted datasets like Fashion MNIST.



---

## 🛠️ Implementation Workflow

1.  **Library Integration:** The process begins by importing required libraries, including TensorFlow, Keras, and the EfficientNet model.
2.  **Data Preprocessing:** The Fashion MNIST dataset is loaded and prepared. This involves adjusting the $28 \times 28$ grayscale images to be compatible with the CNN input pipeline (e.g., resizing or repeating channels).
3.  **Base Model Initialization:** We initialize **EfficientNetB0** without its original "head" (the top classification layer specific to ImageNet). In this configuration, the model acts as a **Feature Extractor (Özellik Çıkarıcı)**.
4.  **Building the Custom Top Layer:** Since the base model only extracts features, a custom classification head must be added for our 10 fashion classes. This typically includes:
    * **Global Average Pooling:** Reducing the spatial dimensions of the features.
    * **Dense (Hidden) Layers:** Fully connected layers to learn fashion-specific patterns.
    * **Softmax Output Layer:** To provide probability scores for the 10 categories.
5.  **Head Training:** Initially, we only train the "head" (top layers) of the model while keeping the EfficientNetB0 base **frozen**. This ensures the pre-trained features are not lost during initial training.
6.  **Fine-Tuning (Optional):** Once the top layers are stabilized, the base model can be **"unfrozen" (çözmek)**. Training the entire model together can lead to higher precision.
7.  **Model Evaluation:** Finally, the model is evaluated on the test set to determine its **Generalization (Genelleme)** capability on unseen data.

---

## 📊 Summary of Advantages

| Feature | Basic ANN (Artificial Neural Network) | CNN with Transfer Learning |
| :--- | :--- | :--- |
| **Input Handling** | Flattened 1D Vector | 2D Spatial Structure |
| **Feature Extraction** | Manual / Limited | Automatic via Convolutional Filters |
| **Data Requirement** | High for complex patterns | Lower (leverages pre-trained features) |
| **Computation** | Fast but less accurate | Highly efficient with EfficientNet-B0 |

---


# Implementation Notes & Computational Requirements

---

### 🖥️ Hardware Constraints & Operational Logic

Due to the resource-intensive nature of **Deep Transfer Learning** and the **Stochastic Optimization** processes involved in this project, certain high-compute modules were structurally verified but not executed to full completion on local hardware. The following constraints were identified during the development lifecycle:

#### 1. Memory Overhead (RAM/VRAM Utilization)
The requirement to upscale **60,000 Fashion MNIST images** from their native $28 \times 28$ grayscale format to $224 \times 224$ RGB tensors created a significant data bottleneck. This transformation resulted in a massive expansion of the feature space, which exceeded local memory thresholds during batch processing.



#### 2. Compute Latency & Processing Bottlenecks
In the absence of a high-performance **NVIDIA GPU** (utilizing CUDA cores), the estimated time-per-epoch for advanced stages—specifically **Strategic Fine-Tuning** and **Automated Hyperparameter Tuning** (via KerasTuner)—was deemed prohibitive for iterative prototyping. 

#### 3. Scientific Validation Strategy
Despite these hardware limitations, the architectural logic and preprocessing pipelines have been rigorously audited for **Scientific Soundness**. The results currently reflected in the project logs represent **Proof-of-Concept (PoC)** executions performed on stratified data subsets. This ensures that the methodology is valid and ready for full-scale deployment on cloud-based GPU clusters (e.g., AWS P3 instances or Google Colab Pro).

---

### 📊 Computational Resource Analysis Table

| Analysis Area | Problems & Components | Technical Detail & Importance | Solution Methods | Tools & Tests |
| :--- | :--- | :--- | :--- | :--- |
| **Data Scaling** | Tensor Dimension Expansion | Upscaling $28^2 \to 224^2$ increases pixel count by $64\times$. | Batch-wise Generators | `tf.data.Dataset` |
| **Optimization** | Hyperparameter Search | High dimensionality in search space (LR, Dropout, Neurons). | Bayesian Optimization | **KerasTuner / Optuna** |
| **Hardware** | Lack of Parallelization | CPU-bound training leads to exponential latency. | GPU Acceleration | **CUDA / cuDNN** |
| **Integrity** | Structural Auditing | Ensuring logic holds before full-scale compute spend. | Stratified Subsampling | `scikit-learn` |

---

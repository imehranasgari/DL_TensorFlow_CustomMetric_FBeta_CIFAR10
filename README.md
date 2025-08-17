# 📊 CIFAR-10 Image Classification with Custom F-Beta Metric

## 1️⃣ Problem Statement and Goal of Project

The goal of this project is to classify images from the **CIFAR-10 dataset** into 10 categories while evaluating performance with a **custom F-Beta score** metric.
This approach goes beyond simple accuracy, providing a more balanced evaluation that considers both precision and recall — essential in imbalanced or cost-sensitive classification problems.

## 2️⃣ Solution Approach

* **Dataset Preparation**:

  * Loaded CIFAR-10 dataset from `tensorflow.keras.datasets`.
  * Normalized pixel values to \[0, 1].
  * One-hot encoded class labels.
* **Custom Metric**:

  * Implemented a `FBetaScore` class by subclassing `tf.keras.metrics.Metric`.
  * Uses built-in `Precision` and `Recall` metrics internally.
  * Prevents division-by-zero errors with epsilon handling.
* **Model Architecture**:

  * Convolutional layers:

    * Conv2D(32) → MaxPooling2D
    * Conv2D(64) → MaxPooling2D
    * Conv2D(128) → MaxPooling2D
  * Fully connected layers: Three Dense(50, ReLU) with Dropout(0.5).
  * Output layer: Dense(10, softmax).
* **Training**:

  * Optimizer: Adam
  * Loss: categorical crossentropy
  * Metrics: Accuracy + custom F-Beta score
  * Callback: EarlyStopping on validation loss with patience=3.
* **Evaluation**:

  * Computed accuracy and F-Beta score on test set.
  * Plotted F-Beta progression across epochs for both training and validation sets.

## 3️⃣ Technologies & Libraries

* **Deep Learning Framework**: TensorFlow / Keras
* **Core Libraries**: NumPy, Matplotlib
* **Dataset Source**: CIFAR-10 (`tensorflow.keras.datasets`)

## 4️⃣ Dataset Description

* **Name**: CIFAR-10
* **Samples**: 60,000 color images (32×32) in 10 classes, with 6,000 images per class.
* **Split**: 50,000 training, 10,000 test.
* **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## 5️⃣ Installation & Execution Guide

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install tensorflow numpy matplotlib

# Open and run the notebook
jupyter notebook project02_fbeta.ipynb
```

## 6️⃣ Key Results / Performance

* Model achieves competitive accuracy and balanced F-Beta scores.
* F-Beta plots reveal convergence trends and potential class balance insights.
* EarlyStopping helps prevent overfitting while reducing training time.

## 7️⃣ Screenshots / Sample Output

*(Refer to notebook for actual plots and logs)*

* Model summary output from `model.summary()`.
* Training/validation F-Beta score progression plot.

## 8️⃣ Additional Learnings / Reflections

* Implementing custom metrics in TensorFlow deepens understanding of the `tf.keras.metrics` API.
* F-Beta score is highly useful when precision and recall need to be balanced differently.
* Combining EarlyStopping with custom metrics allows for robust, efficient training.

---

## 👤 Author

**Mehran Asgari**
📧 [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)
🐙 [https://github.com/imehranasgari](https://github.com/imehranasgari)

---

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

---

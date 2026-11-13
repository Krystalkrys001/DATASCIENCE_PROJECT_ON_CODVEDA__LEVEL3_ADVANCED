# Level 3 — Task 3: Feed-Forward Neural Network & Hyperparameter Tuning (MNIST)

## Project Overview
A Multi-Layer Perceptron (MLP) implemented in **TensorFlow 2.21 / Keras** to classify 70,000 handwritten digit images (`MNIST`, classes `0–9`), evaluating training/validation learning dynamics and tuning `learning_rate` and `batch_size` across three controlled experiments.

## Architecture & Preprocessing
* **Input Preprocessing:** Flattened 28×28 grayscale images into 784-element float32 vectors normalized to `[0.0, 1.0]` (`60,000` train with 10% validation split; `10,000` unseen test images).
* **Network Topology (`235,146` trainable parameters):** `Input(784) → Dense(256, ReLU) → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)`, compiled with Adam and `sparse_categorical_crossentropy` for 12 epochs.

## Hyperparameter Tuning Results (Test Set, N = 10,000)
| Configuration | Learning Rate | Batch Size | Final Val Acc | Test Loss | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Config 1 (Baseline)** | 0.0010 | 128 | **0.9838** | 0.0685 | 0.9812 |
| **Config 2 (High LR)** | 0.0050 | 64 | 0.9715 | 0.1392 | 0.9672 |
| **Config 3 (Tuned Optimal)** | **0.0005** | **64** | 0.9833 | **0.0624** | **0.9817** |

## Key Takeaway
Lowering the Adam learning rate to `0.0005` paired with `batch_size = 64` eliminated the gradient overshoot seen at `LR = 0.005` (`0.1392` loss), achieving smooth convergence and **98.17% unseen test accuracy**.

## Tech Stack
`Python 3.12` • `TensorFlow 2.21` • `Keras` • `NumPy` • `Pandas` • `Matplotlib` • `Seaborn`
# Micrograd From Scratch – Sahasra

This repository implements **automatic differentiation and backpropagation from scratch** in Python, inspired by Andrej Karpathy’s *micrograd*, but written line-by-line without copying.

---

## 📦 Files in the Repository

### `value.py`
A minimal **autodiff engine** that builds a computation graph and performs **reverse-mode backpropagation**.

Features:
- Tracks operations (+, *)
- Stores gradients
- Supports `tanh` activation
- Topological traversal for `.backward()`

### `nn.py`
A tiny neural network library using the `Value` class.

Includes:
- `Neuron` – weights + bias + activation
- `Layer` – multiple neurons
- `MLP` – multi-layer perceptron

### `train.py`
A training script that:
- Creates a simple XOR-style dataset
- Builds a model (`MLP(2, [4, 1])`)
- Runs forward + backward passes
- Updates parameters with gradient descent
- Prints loss every few epochs

---

## 🚀 Run the Code


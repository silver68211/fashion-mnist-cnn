# Fashion-MNIST CNN Training Pipeline
```markdown

A modular and research-oriented deep learning pipeline for Fashion-MNIST classification using TensorFlow / Keras.

This repository provides:

- Multiple CNN architectures
- K-fold cross validation
- Model checkpointing
- Evaluation utilities
- Confusion matrix visualization
- Bayesian hyperparameter tuning (Keras Tuner)

```
## Project Structure


```
fashion-mnist-cnn/

├── config.py        # Central configuration
├── data.py          # Dataset loading and preprocessing
├── models.py        # CNN architectures
├── train_eval.py    # Training and k-fold evaluation
├── metrics_viz.py   # Visualization and metrics utilities
├── tuning.py        # Hyperparameter tuning (Bayesian Optimization)
├── utils_io.py      # File utilities
├── main.py          # Entry point

````

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

```bash
python main.py
```

This will:

1. Load Fashion-MNIST
2. Run k-fold validation
3. Select best fold checkpoint
4. Evaluate on test set
5. Plot training curves

---

## Hyperparameter Tuning

```python
from tuning import run_tuning
best_model = run_tuning(x_train, y_train)
```

---

## Architectures Included

* Baseline CNN
* Modified Padding CNN
* Deep CNN
* Batch-Normalized CNN
* Dropout Regularized CNN

---

## Requirements

* Python 3.9+
* TensorFlow 2.x
* NumPy
* scikit-learn
* matplotlib
* keras-tuner

---



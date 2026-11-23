# Level 1 - Centralized Learning

Trains a neural network on MNIST in a centralized setting.

## Models
- `LeNetWide32` (Primary model, default)
- `LeNet5` (Baseline comparison model, LeCun et al., 1998)

## Files
- `models.py`: Model architectures (`LeNetWide32`, `LeNet5`).
- `train.py`: Loads MNIST data, trains the model, and saves the best model to `best_model.pth`.
- `test.py`: Loads `best_model.pth` and evaluates test accuracy.

## How to Run

Train the model:
```bash
python train.py
```

Test the saved model:
```bash
python test.py
```

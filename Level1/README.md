# Level 1 - Centralized Learning

Trains a CNN on MNIST in a centralized setting.

## Files
- `models.py`: ConvNet model architecture.
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

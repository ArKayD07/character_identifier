# Character Identifier

A PyTorch project for training a CNN-based handwritten character classifier on **EMNIST** or **MNIST**, with a Tkinter GUI (`app.py`) for drawing a character and predicting it live, plus a command-line training and prediction pipeline (`main.py`).

## Key features

- CNN classifier (`EMNISTClassifier` in `main.py`) - two convolutional blocks feeding a small fully-connected head, with dropout for regularization.
- Train on either dataset via a single `--dataset` flag:
  - `emnist` (default) - EMNIST "byclass" split, 62 classes (digits + upper/lowercase letters).
  - `mnist` - standard MNIST digits, 10 classes.
- Automatic dataset download via `torchvision.datasets` (no manual data prep needed).
- Early stopping and a `ReduceLROnPlateau` learning-rate scheduler during training.
- Tkinter drawing app (`app.py`) - draw a character with the mouse and get a live prediction with confidence.
- CLI image prediction (`main.py predict`) - run inference on a saved image file (e.g. a photo of handwriting).

## Repo contents

| File | Purpose |
| --- | --- |
| `main.py` | Training and prediction CLI. Defines the model, data loading, training loop, and the photo-based `predict` command. |
| `app.py` | Tkinter app - draw a character on a canvas and click Predict. |
| `requirements.txt` | Python dependencies. |
| `mnist_character_model.pth` | Pretrained checkpoint (MNIST, 10 classes) used by `app.py` by default. |
| `emnist_test_model.pth` | Additional saved checkpoint from EMNIST testing/training. |

Trained model files (`*.pth`) are checkpoints containing both the model weights and the list of class labels, so they're self-contained - no need to know which dataset a checkpoint came from ahead of time.

## Setup

1. Clone or download the repo, then open a terminal in the `character_identifier` folder.

2. (Recommended) Create and activate a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:

   ```
   python -m pip install -r requirements.txt
   ```

   `requirements.txt` pins: `torch`, `torchvision`, `pillow`, and `numpy`. `tkinter` ships with standard Python on Windows, so no extra install is usually needed for the GUI.

## Training a model

Train on **MNIST**:

```
python main.py train --dataset mnist --epochs 10 --model-path mnist_character_model.pth
```

Train on **EMNIST** (default dataset, "byclass" split):

```
python main.py train --dataset emnist --epochs 3 --model-path emnist_character_model.pth
```

Useful flags for the `train` command:

- `--dataset {emnist,mnist}` - which dataset to train on (default: `emnist`).
- `--data-dir` - where dataset files are downloaded/cached (default: `data`; e.g. `data/MNIST/raw` or `data/EMNIST/raw`).
- `--model-path` - where to save the checkpoint. If omitted, defaults to `<dataset>_character_model.pth`.
- `--epochs` - number of training epochs (default: `3`).
- `--batch-size` - training batch size (default: `64`).
- `--limit-train`, `--limit-test` - optionally cap the number of samples used, handy for a quick smoke test.

The first run for a given dataset will download it automatically via `torchvision`.

## Using the drawing app

`app.py` loads a saved checkpoint and lets you draw a character to classify:

```
python app.py
```

- Draw by holding the left mouse button and dragging on the black canvas.
- Click **Predict** to see the predicted character and confidence.
- Click **Clear** to reset the canvas.
- Click **Load Model** to reload the checkpoint from disk (useful after retraining).

By default `app.py` loads `mnist_character_model.pth` (set via the `MODEL_PATH` constant near the top of the file). To use an EMNIST-trained checkpoint instead, either save your EMNIST training run to that same filename, or edit `MODEL_PATH` in `app.py` to point at your checkpoint (e.g. `emnist_character_model.pth`).

## Predicting from an image file

For a saved image (e.g. a photo of handwriting) rather than the live canvas:

```
python main.py predict --model-path mnist_character_model.pth --image path\to\image.png
```

`--model-path` defaults to `emnist_character_model.pth` if not specified, so pass it explicitly when using an MNIST checkpoint. The image is auto-contrasted, resized to 28x28, and inverted (assuming dark strokes on a light background, like a photo) before inference.

## Notes

- Both training datasets use grayscale 28x28 inputs normalized with mean `0.1307` / std `0.3081`.
- The canvas in `app.py` is already drawn white-on-black (matching the training format), so it is *not* inverted before inference - only the photo-based `main.py predict` path inverts, since photos are typically dark ink on a light background.
- `python main.py train ...` prints an environment check (Python/Torch version, CUDA availability) before training starts.
- Training saves the best checkpoint (lowest test loss) as it goes, and stops early if test loss hasn't improved for 3 epochs.

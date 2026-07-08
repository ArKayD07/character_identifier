# EMNIST Character Identifier

A small PyTorch project for training an EMNIST character classifier and predicting drawn characters with a simple Tkinter app.

## Files

- `main.py` - training and prediction CLI for an EMNIST-based classifier.
- `app.py` - Tkinter drawing app that loads the saved model and predicts the drawn character.
- `requirements.txt` - Python dependencies.

## Setup

1. Open a terminal in `character_identifier`:
   ```powershell
   cd character_identifier
   ```

2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

## Train the model

Run training to download EMNIST and save a model file:

```powershell
python main.py train --epochs 3 --batch-size 64 --model-path emnist_character_model.pth
```

You can use `--limit-train` and `--limit-test` for quick tests.

## Use the drawing app

After training, start the Tkinter app:

```powershell
python app.py
```

- Draw by holding the left mouse button and dragging
- Click `Predict`
- The app will display the predicted character and confidence

## Predict from an image

Use the CLI predictor with a saved model and image file:

```powershell
python main.py predict --model-path emnist_character_model.pth --image path\to\image.png
```

## Notes

- The app expects `emnist_character_model.pth` to exist in the current folder.
- EMNIST data should be downloaded to a directory `data/EMNIST/raw`.
- `tkinter` is included with standard Python on Windows, so no extra install is usually required.

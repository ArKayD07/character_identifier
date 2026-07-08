import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from main import EMNISTClassifier

MODEL_PATH = "emnist_character_model.pth"
CANVAS_SIZE = 280
DRAW_WIDTH = 20


class DrawApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMNIST Character Drawer")
        self.resizable(False, False)

        self.model = None
        self.loaded = False

        self.canvas = tk.Canvas(self, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.canvas_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.canvas_image)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.predict_button = tk.Button(self, text="Predict", width=12, command=self.predict)
        self.predict_button.grid(row=1, column=0, padx=5, pady=(0, 10))

        self.clear_button = tk.Button(self, text="Clear", width=12, command=self.clear_canvas)
        self.clear_button.grid(row=1, column=1, padx=5, pady=(0, 10))

        self.load_button = tk.Button(self, text="Load Model", width=12, command=self.load_model)
        self.load_button.grid(row=1, column=2, padx=5, pady=(0, 10))

        self.status_label = tk.Label(self, text="Draw a character and press Predict.", anchor="w")
        self.status_label.grid(row=2, column=0, columnspan=3, sticky="we", padx=10)

        self.result_label = tk.Label(self, text="Prediction: n/a", font=(None, 12, "bold"), anchor="w")
        self.result_label.grid(row=3, column=0, columnspan=3, sticky="we", padx=10, pady=(0, 10))

        self.load_model(initial=True)

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_stroke(self, event):
        x, y = event.x, event.y
        self.canvas.create_line(self.last_x, self.last_y, x, y, fill="white", width=DRAW_WIDTH, capstyle=tk.ROUND, smooth=True)
        self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=DRAW_WIDTH)
        self.last_x = x
        self.last_y = y

    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_label.config(text="Prediction: n/a")
        self.status_label.config(text="Canvas cleared. Draw a character and press Predict.")

    def load_model(self, initial=False):
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            if initial:
                self.status_label.config(text=f"Model not found: {model_path}. Train the model first.")
            else:
                messagebox.showwarning("Model missing", f"Model file not found:\n{model_path}")
            self.loaded = False
            self.model = None
            return

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            classes = checkpoint["classes"]
            model = EMNISTClassifier(num_classes=len(classes))
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self.model = model
            self.classes = classes
            self.loaded = True
            self.status_label.config(text=f"Model loaded from {model_path}. Draw and predict.")
        except Exception as exc:
            self.loaded = False
            self.model = None
            messagebox.showerror("Load failed", f"Failed to load model:\n{exc}")
            self.status_label.config(text="Failed to load model. See error message.")

    def preprocess_canvas(self):
        image = self.canvas_image.copy()
        image = ImageOps.autocontrast(image)
        image = image.resize((28, 28), resample=Image.Resampling.LANCZOS)
        image = ImageOps.invert(image)
        array = np.array(image, dtype=np.float32) / 255.0
        array = (array - 0.1307) / 0.3081
        tensor = torch.tensor(array).unsqueeze(0).unsqueeze(0)
        return tensor

    def predict(self):
        if not self.loaded or self.model is None:
            messagebox.showwarning("No model", "No trained model is loaded. Train the model first and press Load Model.")
            return

        tensor = self.preprocess_canvas()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, index = torch.max(probabilities, dim=1)

        label = self.classes[index.item()]
        self.result_label.config(text=f"Prediction: {label} ({confidence.item():.1%})")
        self.status_label.config(text="Prediction complete.")


if __name__ == "__main__":
    app = DrawApp()
    app.mainloop()

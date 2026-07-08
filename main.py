import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class EMNISTClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_dataloaders(data_dir: str, batch_size: int, limit_train: int | None, limit_test: int | None):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.EMNIST(
        root=data_dir,
        split="byclass",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="byclass",
        train=False,
        download=True,
        transform=transform,
    )

    if limit_train is not None:
        train_dataset = Subset(train_dataset, list(range(min(limit_train, len(train_dataset)))))
    if limit_test is not None:
        test_dataset = Subset(test_dataset, list(range(min(limit_test, len(test_dataset)))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes
    return train_loader, test_loader, classes


def train_model(data_dir: str, model_path: str, epochs: int, batch_size: int, limit_train: int | None, limit_test: int | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = build_dataloaders(data_dir, batch_size, limit_train, limit_test)
    num_classes = len(classes)

    model = EMNISTClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        test_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Epoch {epoch + 1}/{epochs} | train_loss={running_loss / len(train_loader):.4f} | "
            f"test_loss={test_loss / len(test_loader):.4f} | accuracy={100 * correct / total:.2f}%"
        )

    torch.save({"model_state": model.state_dict(), "classes": classes}, model_path)
    print(f"Model saved to {model_path}")


def preprocess_image(image_path: str):
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.1307,), (0.3081,))(image)
    return image.unsqueeze(0)


def predict_image(model_path: str, image_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint["classes"]
    model = EMNISTClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, index = torch.max(probabilities, 1)

    predicted_label = classes[index.item()]
    print(f"Predicted character: {predicted_label}")
    print(f"Confidence: {confidence.item():.2%}")


def main():
    parser = argparse.ArgumentParser(description="Train an EMNIST-based character classifier and predict from an image")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model on EMNIST")
    train_parser.add_argument("--data-dir", default="data", help="Directory used for EMNIST data")
    train_parser.add_argument("--model-path", default="emnist_character_model.pth", help="Path for the trained model")
    train_parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    train_parser.add_argument("--limit-train", type=int, default=None, help="Optional limit for the number of training samples")
    train_parser.add_argument("--limit-test", type=int, default=None, help="Optional limit for the number of test samples")

    predict_parser = subparsers.add_parser("predict", help="Predict a character from an image")
    predict_parser.add_argument("--model-path", default="emnist_character_model.pth", help="Path to the saved model")
    predict_parser.add_argument("--image", required=True, help="Path to the input image")

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            data_dir=args.data_dir,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            limit_train=args.limit_train,
            limit_test=args.limit_test,
        )
    elif args.command == "predict":
        predict_image(model_path=args.model_path, image_path=args.image)


if __name__ == "__main__":
    main()
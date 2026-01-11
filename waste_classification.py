import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def main() -> None:
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    full_dataset = datasets.ImageFolder(root="./dataset", transform=train_transforms)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Classes:", full_dataset.classes)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False

    num_classes = len(full_dataset.classes)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.last_channel, num_classes),
    )
    model = model.to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

    _ = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=20,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    main()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 20,
) -> dict[str, list[float]]:
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for _ in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples if total_samples else 0.0
        epoch_acc = running_corrects / total_samples if total_samples else 0.0
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_samples += labels.size(0)

        epoch_val_loss = val_loss / val_samples if val_samples else 0.0
        epoch_val_acc = val_corrects / val_samples if val_samples else 0.0
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

    return history

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights


def main() -> None:
    data_dir = "./dataset"

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = datasets.ImageFolder(root=data_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_split, val_split = random_split(full_dataset, [train_size, val_size])

    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    train_subset = Subset(train_dataset, train_split.indices)
    val_subset = Subset(val_dataset, val_split.indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    print("Class names:", full_dataset.classes)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    num_classes = len(full_dataset.classes)
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.last_channel, num_classes),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(model)


if __name__ == "__main__":
    main()

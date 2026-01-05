import torch
from torchvision import datasets, transforms
import os

def get_mnist_loaders(batch_size=64, test_batch_size=1000, data_dir='./data'):
    """
    Returns train_loader, val_loader (using test set as val for MVP simplicity).
    """
    # Simple transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, val_loader

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(config):
    """
    Create a data loader for the specified dataset in the config
    
    Args:
        config: Configuration object with dataset settings
        
    Returns:
        DataLoader for the specified dataset
    """
    if config.dataset == "CIFAR10":
        # For CIFAR10, we need to normalize with the appropriate mean and std for RGB
        # Also resize from 32x32 to 16x16 for faster training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(
            root='data',
            train=True,
            download=True,
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        return train_loader
    elif config.dataset == "MNIST":
        # Also resize MNIST images to match the configured image size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.MNIST(
            root='data',
            train=True,
            download=True,
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        return train_loader
    else:
        raise ValueError(f"Dataset {config.dataset} not supported")

def get_real_images(config, num_samples=100):
    """
    Get a batch of real images from the dataset for evaluation
    
    Args:
        config: Configuration object with dataset settings
        num_samples: Number of images to retrieve
        
    Returns:
        Tensor of real images
    """
    if config.dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.CIFAR10(
            root='data',
            train=False,  # Use test set for evaluation
            download=True,
            transform=transform
        )
        
    elif config.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        dataset = datasets.MNIST(
            root='data',
            train=False,  # Use test set for evaluation
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Dataset {config.dataset} not supported for evaluation")
    
    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=True
    )
    
    # Get a batch of images
    images, _ = next(iter(dataloader))
    return images

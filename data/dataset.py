import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os
import requests
from tqdm import tqdm
import gdown
from PIL import Image

def download_lsun(data_dir, category='bedroom'):
    """
    Download LSUN dataset for a specific category
    
    Args:
        data_dir: Directory to save the dataset
        category: Category to download (default: bedroom)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for the bedroom category
    url = "http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip"
    zip_file = os.path.join(data_dir, f"{category}_train_lmdb.zip")
    
    if not os.path.exists(zip_file):
        print(f"Downloading LSUN {category} dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_file, 'wb') as f, tqdm(
            desc=zip_file,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    # Extract only if the lmdb directory doesn't exist
    lmdb_dir = os.path.join(data_dir, f"{category}_train_lmdb")
    if not os.path.exists(lmdb_dir):
        print(f"Extracting {zip_file}...")
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    print("LSUN dataset ready!")
    return lmdb_dir

def download_celeba_hq(data_dir):
    """
    Download CelebA-HQ dataset using StarGAN v2's approach
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # StarGAN v2's CelebA-HQ download URL
    url = "https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=1"
    zip_file = os.path.join(data_dir, "celeba_hq.zip")
    
    if not os.path.exists(os.path.join(data_dir, "celeba_hq")):
        print("Downloading CelebA-HQ dataset from StarGAN v2...")
        gdown.download(url, zip_file, quiet=False)
        
        print("Extracting CelebA-HQ dataset...")
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("CelebA-HQ dataset ready!")
    else:
        print("CelebA-HQ dataset already exists!")

class CelebAHQDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        # Split between train and validation (90% train, 10% val)
        self.image_paths = sorted([f for f in os.listdir(os.path.join(root, "celeba_hq")) if f.endswith('.jpg')])
        split_idx = int(len(self.image_paths) * 0.9)
        self.image_paths = self.image_paths[:split_idx] if train else self.image_paths[split_idx:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "celeba_hq", self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return 0 as label since we don't use it

def download_afhq(data_dir):
    """Download AFHQ dataset using StarGAN v2's direct link"""
    os.makedirs(data_dir, exist_ok=True)
    
    # StarGAN v2's AFHQ download URL (v2 version with better quality)
    url = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"
    zip_file = os.path.join(data_dir, "afhq_v2.zip")
    
    if not os.path.exists(os.path.join(data_dir, "afhq")):
        print("Downloading AFHQ dataset from StarGAN v2...")
        try:
            import gdown
            gdown.download(url, zip_file, quiet=False)
            
            print("Extracting AFHQ dataset...")
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Clean up zip file
            os.remove(zip_file)
            print("AFHQ dataset ready!")
        except Exception as e:
            print(f"Error downloading AFHQ dataset: {e}")
            raise
    else:
        print("AFHQ dataset already exists!")

def get_data_loader(config, image_size=None):
    """
    Create a data loader for the specified dataset in the config
    
    Args:
        config: Configuration object with dataset settings
        image_size: Optional specific image size to use (overrides config.image_size)
        
    Returns:
        DataLoader for the specified dataset
    """
    # Use provided image_size or fall back to config.image_size
    target_size = image_size if image_size is not None else config.image_size
    
    if config.dataset == "CelebA":
        # CelebA-HQ specific transforms
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        try:
            # Download CelebA-HQ if needed
            download_celeba_hq(config.data_dir)
            
            # Create dataset
            full_dataset = CelebAHQDataset(
                root=config.data_dir,
                train=True,
                transform=transform
            )
            
            # Take only the first 5000 samples for faster training
            num_samples = min(5000, len(full_dataset))
            indices = np.arange(num_samples)
            train_dataset = Subset(full_dataset, indices)
            print(f"Using {num_samples} samples from CelebA-HQ dataset")
            
        except Exception as e:
            print(f"Error loading CelebA-HQ dataset: {e}")
            raise
            
    elif config.dataset == "CIFAR10":
        # For CIFAR10, we need to normalize with the appropriate mean and std for RGB
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((target_size, target_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(
            root='data',
            train=True,
            download=True,
            transform=transform
        )
        
    elif config.dataset == "MNIST":
        # Also resize MNIST images to match the configured image size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((target_size, target_size)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.MNIST(
            root='data',
            train=True,
            download=True,
            transform=transform
        )
        
    elif config.dataset == "LSUN":
        # First, ensure LSUN dataset is downloaded and extracted
        lmdb_dir = download_lsun(config.data_dir, config.high_res_category)
        
        # LSUN specific transforms
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        try:
            full_dataset = datasets.LSUN(
                root=config.data_dir,
                classes=[config.high_res_category],
                transform=transform
            )
            
            # Take only the first 5000 samples
            num_samples = min(5000, len(full_dataset))
            indices = np.arange(num_samples)
            train_dataset = Subset(full_dataset, indices)
            print(f"Using {num_samples} samples from LSUN dataset")
            
        except Exception as e:
            print(f"Error loading LSUN dataset: {e}")
            raise
        
    elif config.dataset == "AFHQ":
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        try:
            # Create dataset directly from the train directory
            dataset_path = os.path.join(config.data_dir, 'train', config.afhq_category)
            if not os.path.exists(dataset_path):
                raise ValueError(f"AFHQ category '{config.afhq_category}' not found in {dataset_path}")
            
            full_dataset = datasets.ImageFolder(
                root=os.path.join(config.data_dir, 'train'),
                transform=transform
            )
            
            # Filter for specific category
            indices = [i for i, (path, label) in enumerate(full_dataset.samples) 
                      if config.afhq_category in path]
            train_dataset = Subset(full_dataset, indices)
            
            print(f"Using {len(train_dataset)} samples from AFHQ dataset ({config.afhq_category} category)")
            return DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
        except Exception as e:
            print(f"Error loading AFHQ dataset: {e}")
            raise
        
    else:
        raise ValueError(f"Dataset {config.dataset} not supported")
    
    # Create and return the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader

def get_real_images(config, num_samples=100):
    """
    Get a batch of real images from the dataset for evaluation
    
    Args:
        config: Configuration object with dataset settings
        num_samples: Number of images to retrieve
        
    Returns:
        Tensor of real images
    """
    if config.dataset == "CelebA":
        transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        try:
            # Use validation split for evaluation
            dataset = CelebAHQDataset(
                root=config.data_dir,
                train=False,
                transform=transform
            )
            
            # Take a subset for evaluation
            num_samples_total = min(num_samples, len(dataset))
            indices = np.arange(num_samples_total)
            dataset = Subset(dataset, indices)
            print(f"Using {num_samples} samples from CelebA-HQ dataset for evaluation")
            
        except Exception as e:
            print(f"Error loading CelebA-HQ dataset for evaluation: {e}")
            raise
            
    elif config.dataset == "CIFAR10":
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
        
    elif config.dataset == "LSUN":
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # For LSUN evaluation, we use a subset of the training set
        full_dataset = datasets.LSUN(
            root=config.data_dir,
            classes=[config.high_res_category],
            transform=transform
        )
        
        # Take only the first 5000 samples for consistency
        num_samples_total = min(5000, len(full_dataset))
        indices = np.arange(num_samples_total)
        dataset = Subset(full_dataset, indices)
        print(f"Using {num_samples} samples from LSUN dataset for evaluation")
        
    elif config.dataset == "AFHQ":
        transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        try:
            # Use validation set for evaluation
            dataset_path = os.path.join(config.data_dir, 'afhq', 'val', config.afhq_category)
            full_dataset = datasets.ImageFolder(
                root=os.path.join(config.data_dir, 'afhq', 'val'),
                transform=transform
            )
            
            # Filter for specific category if specified
            if hasattr(config, 'afhq_category'):
                indices = [i for i, (_, label) in enumerate(full_dataset.samples) 
                          if config.afhq_category in full_dataset.samples[i][0]]
                full_dataset = Subset(full_dataset, indices)
            
            # Take a subset for evaluation
            num_samples_total = min(num_samples, len(full_dataset))
            indices = np.arange(num_samples_total)
            dataset = Subset(full_dataset, indices)
            print(f"Using {num_samples} samples from AFHQ dataset for evaluation ({config.afhq_category} category)")
            
        except Exception as e:
            print(f"Error loading AFHQ dataset for evaluation: {e}")
            raise
        
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

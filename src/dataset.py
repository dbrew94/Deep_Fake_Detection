# src/dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection.
    Loads sequences of frames from video clips.
    """

    def __init__(self,
                 frames_dir,
                 sequence_length=10,
                 transform=None):
        self.frames_dir      = frames_dir
        self.sequence_length = sequence_length
        self.transform       = transform
        self.samples         = []
        self.labels          = []

        real_dir = os.path.join(frames_dir, 'real')
        self._load_samples(real_dir, label=0)

        fake_dir = os.path.join(frames_dir, 'fake')
        self._load_samples(fake_dir, label=1)

        print(f"Dataset loaded:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Real samples:  {self.labels.count(0)}")
        print(f"  Fake samples:  {self.labels.count(1)}")

    def _load_samples(self, directory, label):
        if not os.path.exists(directory):
            print(f"WARNING: Directory not found: {directory}")
            return

        for video_name in os.listdir(directory):
            video_dir = os.path.join(directory, video_name)
            if not os.path.isdir(video_dir):
                continue

            frames = sorted([
                f for f in os.listdir(video_dir)
                if f.endswith('.jpg')
            ])

            if len(frames) >= self.sequence_length:
                frame_paths = [
                    os.path.join(video_dir, f)
                    for f in frames[:self.sequence_length]
                ]
                self.samples.append(frame_paths)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]
        label       = self.labels[idx]

        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        frames = torch.stack(frames)
        label  = torch.tensor(label, dtype=torch.float32)
        return frames, label


def get_transforms(augment=False):
    """
    Get transforms for training and validation.

    Args:
        augment: If True, use stronger augmentation to reduce overfitting.
                 If False, use standard augmentation (same as Week 2/3).
    """
    if augment:
        # WEEK 4: Stronger augmentation to combat overfitting
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),                  # Random crop instead of resize
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # WEEK 2/3: Standard augmentation (unchanged)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Validation transform is always the same regardless of augment flag
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def get_dataloaders(frames_dir,
                    batch_size=32,
                    sequence_length=10,
                    train_ratio=0.7,
                    val_ratio=0.15,
                    augment=False):       # NEW parameter
    """
    Create DataLoaders for training.

    Args:
        augment: Pass True to use stronger augmentation (Week 4).
    """
    train_transform, val_transform = get_transforms(augment=augment)

    # Full dataset uses train_transform
    full_dataset = DeepfakeDataset(
        frames_dir=frames_dir,
        sequence_length=sequence_length,
        transform=train_transform
    )

    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Check frames directory.")

    total      = len(full_dataset)
    train_size = int(train_ratio * total)
    val_size   = int(val_ratio   * total)
    test_size  = total - train_size - val_size

    print(f"\nDataset splits:")
    print(f"  Train: {train_size}")
    print(f"  Val:   {val_size}")
    print(f"  Test:  {test_size}")
    if augment:
        print(f"  Augmentation: STRONG (Week 4)")
    else:
        print(f"  Augmentation: STANDARD (Week 2/3)")

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Val and test sets use val_transform (no augmentation)
    val_set.dataset  = DeepfakeDataset(
        frames_dir=frames_dir,
        sequence_length=sequence_length,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader
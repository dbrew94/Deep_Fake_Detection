# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json


def train_one_epoch(model,
                    loader,
                    criterion,
                    optimizer,
                    device,
                    max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for frames, labels in tqdm(loader, desc="Training"):
        frames = frames.to(device)
        labels = labels.long().to(device)   # float32 -> long

        optimizer.zero_grad()
        outputs = model(frames)             # shape [batch, 2]
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping (Week 2 fix)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct     += (predicted == labels).sum().item()
        total       += labels.size(0)
        total_loss  += loss.item()

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Validating"):
            frames = frames.to(device)
            labels = labels.long().to(device)

            outputs = model(frames)
            loss    = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct     += (predicted == labels).sum().item()
            total       += labels.size(0)
            total_loss  += loss.item()

    return total_loss / len(loader), correct / total


def train_model(model,
                train_loader,
                val_loader,
                model_name,
                num_epochs=30,
                learning_rate=0.001,
                device='cuda',
                save_dir='models/',
                patience=None,                  # NEW: early stopping patience
                use_cosine_scheduler=False):    # NEW: cosine vs step LR
    """
    Full training loop with validation and optional early stopping.

    Args:
        patience:              Epochs to wait for val_loss improvement.
                               None disables early stopping.
        use_cosine_scheduler:  True uses CosineAnnealingLR.
                               False uses StepLR (original).
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    if use_cosine_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )

    os.makedirs(save_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'train_acc':  [],
        'val_loss':   [],
        'val_acc':    []
    }

    best_val_acc  = 0.0
    best_val_loss = float('inf')
    patience_ctr  = 0
    stopped_early = False

    print(f"\nTraining {model_name}...")
    print(f"Device:    {device}")
    print(f"Epochs:    {num_epochs}")
    print(f"Patience:  {patience if patience else 'disabled'}")
    print(f"Scheduler: {'Cosine' if use_cosine_scheduler else 'StepLR'}")
    print("-" * 50)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model by val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(
                save_dir, f"{model_name}_best.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved: {save_path}")

        # Early stopping on val_loss
        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr  = 0
            else:
                patience_ctr += 1
                print(
                    f"  -> No val_loss improvement. "
                    f"Patience: {patience_ctr}/{patience}"
                )
                if patience_ctr >= patience:
                    print(f"\n  Early stopping triggered at epoch {epoch+1}")
                    stopped_early = True
                    break

    # Save training history
    history_path = os.path.join(
        save_dir, f"{model_name}_history.json"
    )
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    if stopped_early:
        actual_epochs = len(history['train_loss'])
        print(f"Stopped at epoch:  {actual_epochs}/{num_epochs}")

    return history
from dataset import OverlapTileDataset

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path



def train(dataset_dir = "/data/maestria/datasets/Pinus_Taeda/PinusTaedaV1", tile_size=512, overlap=128, batch_size=2,
          lr=0.001, number_of_epochs=10, tiles = True, logs_dir="runs/unet_experiment", step_size=5, gamma=0.1):
    dataset = OverlapTileDataset(Path(dataset_dir), tile_size=tile_size, overlap=overlap, debug=True, tiles=tiles)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the model
    model = smp.Unet(
        encoder_name="resnet34",  # Choose an encoder (backbone) like resnet34, efficientnet, etc.
        encoder_weights="imagenet",  # Use pretrained weights on ImageNet
        in_channels=3,  # Number of input channels (e.g., 3 for RGB)
        classes=1  # Number of output classes (e.g., 1 for binary segmentation)
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=logs_dir)

    # Ensure the model is moved to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(number_of_epochs):  # Number of epochs
        running_loss = 0.0  # Track total loss for the epoch

        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch

            # Preprocess images
            images = images.permute(0, 3, 1, 2).float() / 255.0  # Normalize to [0, 1]
            labels = labels.float().unsqueeze(1)  # Add channel dimension

            # Move data to GPU if available
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(images)

            # Compute loss
            loss = criterion(predictions, labels)

            # Backpropagation
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            # Accumulate loss
            running_loss += loss.item()

            # Log batch loss to TensorBoard
            writer.add_scalar("Loss/Batch", loss.item(), epoch * len(dataloader) + batch_idx)

        # Step the scheduler
        scheduler.step()

        # Epoch summary
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{number_of_epochs} finished. Avg Loss: {epoch_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Log epoch loss and learning rate to TensorBoard
        writer.add_scalar("Loss/Epoch", epoch_loss, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

    # Close the writer after training
    writer.close()

    return

if __name__ == "__main__":
    logs_dir = "runs/unet_experiment"
    train(logs_dir)


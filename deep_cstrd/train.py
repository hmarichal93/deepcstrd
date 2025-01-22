import os

from deep_cstrd.dataset import OverlapTileDataset, split_dataset
from deep_cstrd.utils import save_batch_with_labels_as_subplots
from deep_cstrd.losses import DiceLoss, Loss
from deep_cstrd.model import segmentation_model

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.detection.mask_rcnn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


def save_config(logs_dir, dataset_root, tile_size, overlap, batch_size, lr, number_of_epochs, tiles, step_size, gamma, loss, augmentation, model_type, debug):
    if Path(logs_dir).exists():
        os.system(f"rm -r {logs_dir}")

    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{logs_dir}/config.txt", "w") as f:
        f.write(f"dataset_root: {dataset_root}\n")
        f.write(f"tile_size: {tile_size}\n")
        f.write(f"overlap: {overlap}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"number_of_epochs: {number_of_epochs}\n")
        f.write(f"tiles: {tiles}\n")
        f.write(f"logs_dir: {logs_dir}\n")
        f.write(f"step_size: {step_size}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"loss: {loss}\n")
        f.write(f"augmentation: {augmentation}\n")
        f.write(f"model_type: {model_type}\n")
        f.write(f"debug: {debug}\n")

def train( dataset_root= Path("/data/maestria/resultados/deep_cstrd/pinus_v1"),
          tile_size=512, overlap=0.1, batch_size=4,
          lr=0.001, number_of_epochs=100, tiles = True, logs_dir="runs/unet_experiment", step_size=20, gamma=0.5,
          loss = Loss.dice , augmentation = False, model_type=segmentation_model.UNET,debug=False):

    save_config(logs_dir, dataset_root, tile_size, overlap, batch_size, lr, number_of_epochs, tiles, step_size, gamma, loss, augmentation, model_type, debug)
    # Create the datasets for train, validation and test
    train_dataset_dir = dataset_root / "train"
    val_dataset_dir = dataset_root / "val"
    test_dataset_dir = dataset_root / "test"
    if not train_dataset_dir.exists() or not val_dataset_dir.exists() or not test_dataset_dir.exists():
        split_dataset(dataset_root, val_size=0.2, test_size=0.2)

    #
    dataset_train = OverlapTileDataset(Path(train_dataset_dir), tile_size=tile_size, overlap=overlap, debug=True, tiles=tiles,
                                       augmentation=augmentation)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = OverlapTileDataset(Path(val_dataset_dir), tile_size=tile_size, overlap=overlap, debug=True, tiles=tiles)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Define the model
    if model_type == segmentation_model.UNET:
        print("UNET")
        model = smp.Unet(
            encoder_name="resnet34",  # Choose an encoder (backbone) like resnet34, efficientnet, etc.
            encoder_weights="imagenet",  # Use pretrained weights on ImageNet
            in_channels=3,  # Number of input channels (e.g., 3 for RGB)
            classes=1  # Number of output classes (e.g., 1 for binary segmentation)
        )
    elif model_type == segmentation_model.UNET_PLUS_PLUS:
        print("UNET++")
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",  # Choose an encoder (backbone) like resnet34, efficientnet, etc.
            encoder_weights="imagenet",  # Use pretrained weights on ImageNet
            in_channels=3,  # Number of input channels (e.g., 3 for RGB)
            classes=1  # Number of output classes (e.g., 1 for binary segmentation)
        )
    elif model_type == segmentation_model.MASK_RCNN:
        print("MASK RCNN")
        model = torchvision.models.detection.mask_rcnn.MaskRCNN(backbone="resnet50", num_classes=1, pretrained=True)

    else:
        raise ValueError("Invalid model type")

    criterion = DiceLoss() if loss == Loss.dice else nn.BCEWithLogitsLoss()
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
        epoch_images_dir = Path(logs_dir) / f"epoch_{epoch}"  # Directory to save images for this epoch
        if epoch_images_dir.exists():
            os.system(f"rm -r {epoch_images_dir}")
        if epoch % step_size == 0 and epoch>0:
            epoch_images_dir.mkdir(parents=True, exist_ok=True)
            epoch_batch_images_dir = epoch_images_dir / "train"
            epoch_batch_images_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, batch in enumerate(dataloader_train):
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
            writer.add_scalar("Loss/Batch", loss.item(), epoch * len(dataloader_train) + batch_idx)
            # Example after getting predictions
            if epoch % step_size == 0 and epoch>0:
                # save_batch_with_labels_as_subplots(
                #     images,
                #     labels,
                #     predictions,
                #     output_path= epoch_batch_images_dir / f"{batch_idx}.png",
                #     batch_size=batch_size
                # )
                #save model
                torch.save(model.state_dict(), f"{epoch_images_dir}/latest_model.pth")

        # Step the scheduler
        scheduler.step()

        # Epoch summary
        epoch_loss = running_loss / len(dataloader_train)
        print(f"Epoch {epoch + 1}/{number_of_epochs} finished. Avg Loss: {epoch_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Log epoch loss and learning rate to TensorBoard
        writer.add_scalar("Loss/Epoch", epoch_loss, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        # Validation loop
        epoch_batch_images_dir = epoch_images_dir / "val"
        if epoch % step_size == 0 and epoch>0 or epoch == number_of_epochs-1:
            epoch_batch_images_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for batch_idx, batch in enumerate(dataloader_val):
                images, labels = batch

                # Preprocess images
                images = images.permute(0, 3, 1, 2).float() / 255.0
                labels = labels.float().unsqueeze(1)

                # Move data to GPU if available
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                predictions = model(images)

                # Compute loss
                loss = criterion(predictions, labels)

                # Accumulate loss
                running_loss += loss.item()

                # Log batch loss to TensorBoard
                writer.add_scalar("Val Loss/Batch", loss.item(), epoch * len(dataloader_val) + batch_idx)

                # Example after getting predictions
                if debug and epoch % step_size == 0 and (epoch>0 or epoch == number_of_epochs-1) :
                    save_batch_with_labels_as_subplots(
                        images,
                        labels,
                        predictions,
                        output_path= epoch_batch_images_dir / f"val_{batch_idx}.png",
                        batch_size=batch_size
                    )


            # Epoch summary
            epoch_loss = running_loss / len(dataloader_val)
            print(f"Validation Loss: {epoch_loss:.4f}")
            writer.add_scalar("Val Loss/Epoch", epoch_loss, epoch)


        model.train()


    #save model
    torch.save(model.state_dict(), f"{logs_dir}/model.pth")
    # Close the writer after training
    writer.close()

    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a U-Net model for image segmentation')
    parser.add_argument('--dataset_dir', type=str, default="/data/maestria/resultados/deep_cstrd/pinus_v1/",
                        help='Path to the dataset directory')

    parser.add_argument('--logs_dir', type=str, default="runs/pinus_v1_40_train_12_val")
    #load rest of parameter from config file
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    parser.add_argument("--augmentation", type=bool, default=False, help="Apply augmentation to the dataset")
    parser.add_argument("--model_type", type=int, default=segmentation_model.UNET, help="Type of model to use")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    args = parser.parse_args()

    train(dataset_root=Path(args.dataset_dir), logs_dir=args.logs_dir, augmentation= args.augmentation,
          model_type=args.model_type, debug=args.debug)


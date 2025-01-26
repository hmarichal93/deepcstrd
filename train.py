import os

from deep_cstrd.dataset import load_datasets
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

def configure_optimizer(model, lr, number_of_epochs, step_size = None, gamma = None):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Define the learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, number_of_epochs, eta_min=lr / 100)
    return optimizer, scheduler


def forward_step(model, criterion, device, batch):
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
    return loss

def train_one_epoch(model, device, dataloader_train, optimizer, criterion, scheduler):
    model.train()
    running_loss = 0.0  # Track total loss for the epoch
    for batch_idx, batch in enumerate(dataloader_train):
        optimizer.zero_grad()  # Clear previous gradients
        # Forward pass
        loss  = forward_step(model, criterion, device, batch)
        # Backpropagation
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        # Accumulate loss
        running_loss += loss.item()

    # Step the scheduler
    scheduler.step()

    # Epoch summary
    epoch_loss = running_loss / len(dataloader_train)
    return epoch_loss

class Logger:
    def __init__(self, logs_dir):
        self.writer = SummaryWriter(log_dir=logs_dir)
        self.epoch_message = ""
        self.epoch = 0

    def on_training_epoch_end(self, epoch, epoch_loss, number_of_epochs):
        self.epoch_message+= f"Epoch {epoch}/{number_of_epochs} | Train Loss: {epoch_loss:.4f} | "
        self.writer.add_scalar("TrainLoss/Epoch", epoch_loss, epoch)

    def on_validation_epoch_end(self, epoch, epoch_loss, lr ):
        self.epoch_message+= f"Val Loss: {epoch_loss:.4f} | lr: {lr:.6f}"
        self.writer.add_scalar("Val Loss/Epoch", epoch_loss, epoch)
        self.writer.add_scalar("lr",lr, epoch)


    def on_epoch_end(self, epoch ):
        self.epoch = epoch + 1
        print(self.epoch_message)
        self.epoch_message = ""

    def on_batch_end(self, logs, batch_idx, total_batches):
        pass

def eval_one_epoch(model, device, dataloader_val, criterion):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader_val):
            loss = forward_step(model, criterion, device, batch)
            # Accumulate loss
            running_loss += loss.item()

    return running_loss / len(dataloader_val)

def load_model(model_type, model_dir):
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

    # Ensure the model is moved to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    torch.cuda.empty_cache()

    if Path(f"{model_dir}/best_model.pth").exists():
        model.load_state_dict(torch.load(f"{model_dir}/best_model.pth"))

    return model, device


def set_output_directory(dataset_root= Path("/data/maestria/resultados/deep_cstrd/pinus_v1"),
          tile_size=512, overlap=0.1, batch_size=4,
          lr=0.001, number_of_epochs=100, tiles = True, logs_dir="runs/unet_experiment", step_size=20, gamma=0.5,
          loss = Loss.dice , augmentation = False, model_type=segmentation_model.UNET,debug=False):
    logs_dir = Path(logs_dir)
    dataset_name = Path(dataset_root).name
    logs_name = f"{dataset_name}_epochs_{number_of_epochs}_tile_{int(tile_size)}_batch_{batch_size}_step_{step_size}"
    logs_dir = str(logs_dir / logs_name)
    save_config(logs_dir, dataset_root, tile_size, overlap, batch_size, lr, number_of_epochs, tiles, step_size, gamma, loss, augmentation, model_type, debug)
    return logs_dir




def training(dataset_root= Path("/data/maestria/resultados/deep_cstrd/pinus_v1"),
             tile_size=512, overlap=0.1, batch_size=4,
             lr=0.001, number_of_epochs=100, tiles = True, logs_dir="runs/unet_experiment", step_size=20, gamma=0.5,
             loss = Loss.dice, augmentation = False, model_type=segmentation_model.UNET, debug=False,
             min_running_loss = 1000000, best_epoch = 0):

    logs_dir = set_output_directory(dataset_root, tile_size, overlap, batch_size, lr, number_of_epochs, tiles, logs_dir,
                                    step_size, gamma, loss, augmentation, model_type, debug)

    dataloader_train, dataloader_val = load_datasets(dataset_root, tile_size, overlap, batch_size, augmentation)

    criterion = DiceLoss() if loss == Loss.dice else nn.BCEWithLogitsLoss()
    model, device = load_model(model_type, logs_dir)
    optimizer, scheduler = configure_optimizer(model, lr, number_of_epochs, step_size, gamma)
    logger = Logger(logs_dir)

    for epoch in range(number_of_epochs):

        epoch_train_loss = train_one_epoch(model, device, dataloader_train, optimizer, criterion, scheduler)

        logger.on_training_epoch_end(epoch, epoch_train_loss, number_of_epochs)

        epoch_val_loss = eval_one_epoch(model, device, dataloader_val, criterion)

        logger.on_validation_epoch_end(epoch, epoch_val_loss, scheduler.get_last_lr()[0])

        logger.on_epoch_end(epoch)

        save_model = epoch_train_loss < min_running_loss
        if save_model:
            min_running_loss = epoch_train_loss
            torch.save(model.state_dict(), f"{logs_dir}/best_model.pth")
            best_epoch = epoch


    #save model
    torch.save(model.state_dict(), f"{logs_dir}/latest_model.pth")
    print(f"Best model in epoch {best_epoch} with loss {min_running_loss}")
    # Close the writer after training
    logger.writer.close()

    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a U-Net model for image segmentation')
    parser.add_argument('--dataset_dir', type=str, default="/data/maestria/resultados/deep_cstrd/pinus_v1/",
                        help='Path to the dataset directory')

    parser.add_argument('--logs_dir', type=str, default="runs/pinus_v1_40_train_12_val")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--tile_size', type=int, default=512, help='Tile size')
    parser.add_argument('--step_size', type=int, default=20, help='Step size for the learning rate scheduler')
    parser.add_argument('--number_of_epochs', type=int, default=40, help='Number of epochs')
    #load rest of parameter from config file
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    parser.add_argument("--augmentation", type=bool, default=False, help="Apply augmentation to the dataset")
    parser.add_argument("--model_type", type=int, default=segmentation_model.UNET, help="Type of model to use")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    args = parser.parse_args()

    training(dataset_root=Path(args.dataset_dir), logs_dir=args.logs_dir, augmentation= args.augmentation,
             model_type=args.model_type, debug=args.debug, batch_size=args.batch_size, tile_size=args.tile_size, step_size=args.step_size,
             number_of_epochs=args.number_of_epochs)


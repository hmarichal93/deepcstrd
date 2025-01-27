import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def save_batch_with_labels_as_subplots(batch,  predictions, output_path="batch_predictions_with_labels.png",
                                       threshold=0.5, batch_size=2):
    """
    Save a batch of images, labels, and predictions as a single subplot.
    Args:
        images (Tensor): Batch of input images (B, C, H, W).
        labels (Tensor): Batch of ground truth masks (B, 1, H, W).
        predictions (Tensor): Batch of predicted masks (B, 1, H, W).
        output_path (str): Path to save the output figure.
        threshold (float): Threshold for binary masks.
    """
    # Ensure predictions are probabilities and convert to binary masks
    probabilities = torch.sigmoid(predictions)  # Convert logits to probabilities
    binary_masks = probabilities > threshold    # Apply threshold
    images, labels = batch

    images_size = images.size(0)
    fig, axes = plt.subplots(images_size, 3, figsize=(15, 5 * batch_size))
    if images_size == 1:
        return
    for i in range(images_size):
        # Convert image, label, and mask to NumPy
        image_np = images[i].cpu().numpy()#.transpose(1, 2, 0)  # Convert to HWC
        image_np = (image_np * 255).astype(np.uint8)           # Rescale to [0, 255]
        #convert to RGB
        #image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        label_np = labels[i].cpu().numpy().squeeze()           # Squeeze channel dimension
        label_np = (label_np * 255).astype(np.uint8)           # Rescale to [0, 255]

        mask_np = binary_masks[i].cpu().numpy().squeeze()      # Squeeze channel dimension

        # Plot original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")

        # Plot ground truth label
        axes[i, 1].imshow(image_np)
        axes[i, 1].imshow(label_np, cmap="jet", alpha=0.5)  # Display label as grayscale
        axes[i, 1].set_title(f"Label {i}")
        axes[i, 1].axis("off")

        # Plot predicted mask
        axes[i, 2].imshow(image_np)
        axes[i, 2].imshow(mask_np, cmap="jet", alpha=0.5)  # Overlay mask
        axes[i, 2].set_title(f"Prediction {i}")
        axes[i, 2].axis("off")

    # Adjust layout and save
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return fig

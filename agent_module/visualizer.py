import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def visualize_images(output_path: str, num_images: int = None) -> str:
    """Load and visualize generated lensing images"""

    images = np.load(output_path, allow_pickle=True)

    # how many to show
    n = min(len(images), num_images or len(images))

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    # handle single image case
    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i].imshow(images[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')

    # get model and substructure from filename
    filename = os.path.basename(output_path)
    parts = filename.split('_')
    title = f"{parts[0]}_{parts[1]} - {parts[2]} substructure"

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    # save the plot
    plot_path = output_path.replace('.npy', '_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {plot_path}")

    return plot_path
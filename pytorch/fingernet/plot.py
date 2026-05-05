import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from pathlib import Path
import glob

def plot_img(ax: plt.Axes, image: np.ndarray):
    """Plot the image on the given axis."""
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_ori_field(ax: plt.Axes, orientation_field: np.ndarray, stride: int = 16):
    """Overlay the orientation field (segments) on the given axis.

    Args:
        ax: Matplotlib axis to draw onto.
        orientation_field: 2-D array of angles in radians.
        stride: Spacing between orientation segments.
    """
    height, width = orientation_field.shape
    # Segment length scaled with stride for legibility
    segment_length = stride * 0.45

    for r in range(stride // 2, height, stride):
        for c in range(stride // 2, width, stride):
            angle = orientation_field[r, c]
            # Skip points with no defined orientation (background where angle = 0)
            if angle != 0:
                dx = segment_length * np.cos(angle)
                dy = segment_length * np.sin(angle)
                # Draw a line through (c, r) along the angle direction
                ax.plot([c - dx, c + dx], [r - dy, r + dy], 'r-', linewidth=1)

def plot_mnt(ax: plt.Axes, minutiae: np.ndarray, r: int = 10):
    """Overlay minutiae (squares + angles) on the given axis.

    Args:
        ax: Matplotlib axis to draw onto.
        minutiae: (N, 4) array with columns [x, y, angle, score].
        r: Segment length that indicates each minutia's angle.
    """
    # Red unfilled squares at (x, y)
    ax.plot(
        minutiae[:, 0],
        minutiae[:, 1],
        'rs',  # 'r' for red, 's' for square
        fillstyle='none',
        markersize=6,
        markeredgewidth=1
    )

    # Draw the orientation segment for each minutia
    for x, y, angle, score in minutiae:
        ax.plot([x, x + r * np.cos(angle)], [y, y + r * np.sin(angle)], 'r-', linewidth=1.5)

def plot_raw_output(
        output: dict,
        orig_img: np.ndarray | None = None,
        figsize: tuple = (20, 6),
        stride: int = 16
):
    orientation_field = output['orientation_field'].squeeze()
    enhanced_image = output['enhanced_image'].squeeze()
    minutiae = output['minutiae']

    if orig_img is None:
        input_image = enhanced_image
    else:
        input_image = orig_img

    # 1x4 subplot grid
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # --- Subplot 1 ---
    ax1 = axes[0]
    plot_img(ax1, orientation_field)
    ax1.set_title("Orientation Field")

    # --- Subplot 2 ---
    ax2 = axes[1]
    plot_img(ax2, enhanced_image)
    ax2.set_title("Enhanced Image")

    # --- Subplot 3 ---
    ax3 = axes[2]
    plot_img(ax3, input_image)
    plot_ori_field(ax3, orientation_field, stride=stride)
    ax3.set_title(f"Orientation Field (Stride: {stride})")

    # --- Subplot 4 ---
    ax4 = axes[3]
    plot_img(ax4, input_image)
    plot_mnt(ax4, minutiae)
    ax4.set_title(f"Detected Minutiae ({len(minutiae)})")

    # Avoid title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])



def plot_output(
    result: dict,
    save_path: str | None = None,
    stride: int = 16,
    figsize: tuple = (20, 6)
):
    """Render a 1x4 figure with the full visualization of inference results.

    Args:
        result (dict): A single dict from the result list returned by
            ``run_inference``. Must contain ``input_path``,
            ``orientation_field``, ``enhanced_image`` and ``minutiae``.
        save_path (str | None): Path to save the figure. If None, displays.
        stride (int): Stride for the orientation-field visualization.
    """
    try:
        # Load the original input image for overlays
        input_image = np.array(Image.open(result['input_path']).convert('L'))
    except FileNotFoundError:
        print(f"Error: input image not found at {result['input_path']}")
        return

    # Pull data from the result dict
    orientation_field = result['orientation_field'].squeeze()
    enhanced_image = result['enhanced_image'].squeeze()
    minutiae = result['minutiae'][0]

    # 1x4 subplot grid
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # --- Subplot 1 ---
    ax1 = axes[0]
    plot_img(ax1, orientation_field)
    ax1.set_title("Orientation Field")

    # --- Subplot 2 ---
    ax2 = axes[1]
    plot_img(ax2, enhanced_image)
    ax2.set_title("Enhanced Image")

    # --- Subplot 3 ---
    ax3 = axes[2]
    plot_img(ax3, input_image)
    plot_ori_field(ax3, orientation_field, stride=stride)
    ax3.set_title(f"Orientation Field (Stride: {stride})")

    # --- Subplot 4 ---
    ax4 = axes[3]
    plot_img(ax4, input_image)
    plot_mnt(ax4, minutiae)
    ax4.set_title(f"Detected Minutiae ({len(minutiae)})")

    # Figure-wide title
    base_name = os.path.basename(result['input_path'])
    fig.suptitle(f"FingerNet results for: {base_name}", fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_from_output_folder(
    output_path: str,
    image_filename: str,
    save_path: str | None = None,
    stride: int = 16,
    degrees: bool = False,
    input_path: str | None = None,
):
    """Plot inference results from the on-disk output folder structure,
    reconstructing the file paths for a specific image.

    Args:
        output_path (str): Path to the top-level results folder (e.g. 'output/').
        image_filename (str): Original image filename (e.g. '101_1.png').
        save_path (str | None): Path to save the figure. If None, displays.
        stride (int): Stride for the orientation-field visualization.
    """
    print(f"INFO: Generating visualization for '{image_filename}' from '{output_path}'...")

    # Use only the basename of whatever the user passed.
    # This way they can pass either '100_1.png' or 'enhanced/100_1.png'.
    image_basename = os.path.basename(image_filename)
    base_name = Path(image_basename).stem

    # --- Reconstruct artifact paths based on the output folder layout ---
    # Candidate directory name patterns for each artifact group.
    enhanced_dirs = ['enh*']
    orientation_dirs = ['ori*', 'orientation*']
    minutiae_dirs = ['mnt*', 'minutiae*']

    def find_file_by_dir_patterns(base_dir: str, dir_patterns: list[str], filename: str):
        """Search for filename inside subdirectories of base_dir matching any of dir_patterns.

        Returns the first full path found or None.
        """
        # Try exact path relative to base_dir first
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            return candidate

        for pat in dir_patterns:
            glob_path = os.path.join(base_dir, pat)
            for match in glob.glob(glob_path):
                if os.path.isdir(match):
                    full = os.path.join(match, filename)
                    if os.path.exists(full):
                        return full
        return None

    enhanced_path = find_file_by_dir_patterns(output_path, enhanced_dirs, image_basename)
    orientation_path = find_file_by_dir_patterns(output_path, orientation_dirs, image_basename)
    minutiae_path = find_file_by_dir_patterns(output_path, minutiae_dirs, f"{base_name}.min")

    # Make sure all required files exist
    for name, path in (('enhanced', enhanced_path), ('orientation', orientation_path), ('minutiae', minutiae_path)):
        if path is None:
            print(f"ERROR: could not locate the {name} file for '{image_basename}' under '{output_path}'.")
            return
        if not os.path.exists(path):
            print(f"ERROR: required file not found: {path}")
            return

    # Load file data
    if input_path is not None:
        display_image = np.array(Image.open(input_path).convert('L'))
    else:
        display_image = np.array(Image.open(enhanced_path).convert('L'))
    orientation_img = np.array(Image.open(orientation_path))
    # orientation_img is stored in degrees in many pipelines; subtract 90 then convert
    orientation_field = np.deg2rad(orientation_img.astype(np.float32) - 90.0)

    minutiae = np.loadtxt(minutiae_path, comments='#')
    if minutiae.ndim == 1 and minutiae.size > 0:  # handle single-minutia case
        minutiae = np.expand_dims(minutiae, 0)
    elif minutiae.size == 0:  # handle no-minutiae case
        minutiae = np.empty((0, 4))

    # Convert .min angles (CCW degrees) to CW radians (matplotlib screen coords)
    minutiae[:, 2] = (-np.deg2rad(minutiae[:, 2])) % (2 * np.pi)

    # --- 3-subplot figure (plotting logic unchanged) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Image
    plot_img(axes[0], display_image)
    axes[0].set_title("Original Image" if input_path is not None else "Enhanced Image")

    # 2. Image + orientation field
    plot_img(axes[1], display_image)
    plot_ori_field(axes[1], orientation_field, stride=stride)
    axes[1].set_title(f"Orientation Field (Stride: {stride})")

    # 3. Image + minutiae
    plot_img(axes[2], display_image)
    plot_mnt(axes[2], minutiae)
    axes[2].set_title(f"Detected Minutiae ({len(minutiae)})")

    # Figure title + save
    fig.suptitle(f"FingerNet results for: {image_filename}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    else:
        # plt.show() can fail in headless environments
        print("WARNING: no save_path given. Plot will not be shown in headless environments.")
        # plt.show()

    plt.close(fig)

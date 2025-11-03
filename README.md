# FingerNet: Unified Deep Network for Fingerprint Minutiae Extraction

## Introduction

**FingerNet** is a deep convolutional neural network designed for fingerprint minutiae extraction that successfully combines traditional fingerprint processing domain knowledge with the advanced representation capabilities of deep learning.

The original paper addresses a critical challenge in fingerprint analysis: while traditional methods using handcrafted features perform well on clean (rolled/slap) fingerprints, they often fail on latent (crime scene) fingerprints due to complex background noise and fuzzy ridges. Conversely, standard deep learning approaches typically fail to incorporate the specialized domain knowledge already established in fingerprint analysis.

The key innovation of FingerNet lies in its two-fold approach:

1. **Transform Traditional Methods into a Network**: The typical fingerprint analysis pipeline (orientation estimation, segmentation, Gabor enhancement, and minutiae extraction) is mathematically transformed into a shallow neural network with fixed weights.

2. **Expand and Train the Network**: This shallow network is then expanded by adding more convolutional layers to enhance its representative power, and critically, the weights are released, making the entire network trainable end-to-end.

This design allows FingerNet to learn how to handle complex background noise from data while still being guided by a structure inspired by proven, traditional fingerprint processing steps. The resulting unified network can simultaneously output the orientation field, segmentation map, enhanced fingerprint, and minutiae maps, demonstrating superior performance over state-of-the-art methods on both latent and slap fingerprint databases.

## About This Implementation

This repository contains an **extended PyTorch implementation** of FingerNet. The original implementation was developed in Python 2.7 with TensorFlow. This version has been completely rewritten for **Python 3.10+** using **PyTorch**, featuring:

- **Modern PyTorch architecture** with optimized inference pipelines
- **Multi-GPU distributed inference** using PyTorch DDP (DistributedDataParallel)
- **Scalable batch processing** for large-scale fingerprint extraction
- **Flexible API** for both programmatic use and command-line interface
- **Optimized performance** with torch.compile and memory-efficient processing strategies

This implementation is designed for production-scale fingerprint analysis, enabling efficient processing of large datasets across multiple GPUs.

## Running the Original TensorFlow Implementation

If you want to run the **original TensorFlow/Python 2.7 implementation** for comparison or legacy purposes, a Docker image is provided.

```bash
# Check if the image already exists
docker image ls -a

# Build the image if it doesn't exist
docker build -t fingernet:legacy .
```

CPU-only Mode

```bash
docker run -it --rm -v "$PWD":/workspace/FingerNet fingernet:legacy
```

With GPU Support

```bash
docker run -it --rm --gpus all -v "$PWD":/workspace/FingerNet fingernet:legacy
```

Once inside the container:

```bash
# Navigate to the source directory
cd src

# Run the demo on provided test images
# Syntax: python train_test_deploy.py <GPU_ID> <mode>
# GPU_ID: 0, 1, 2, ... (use 0 for CPU mode)
# mode: train, test, or deploy
python train_test_deploy.py 0 deploy
```

The demo will process sample images from the `datasets/` directory and output results to `datasets/output/`.

**Note**: The original implementation does not support multi-GPU distributed processing.

## Installation (PyTorch Version)

### Basic Installation

```bash
cd pytorch
pip install .
```

### Development Installation (Editable Mode)

For development with full IDE support (PyLance/VSCode):

```bash
cd pytorch
pip install -e . --config-settings editable_mode=strict
```

This ensures proper type checking and autocompletion in your IDE.

## Usage

### Batch Inference

```python
from fingernet.api import run_inference

# Process a single directory of images
run_inference(
    input_path='./images',
    output_path='./output',
    gpus=1,  # Use single GPU
    batch_size=8
)
```

Multi-GPU Distributed Inference

```python
from fingernet.api import run_inference

# Use 2 GPUs (GPU 0 and GPU 1)
run_inference(
    input_path='./images',
    output_path='./output',
    gpus=2,  # Use first 2 GPUs
    batch_size=16,  # Batch size per GPU
    num_workers=4,  # DataLoader workers per GPU
    compile_model=True,  # Use torch.compile for speedup
    recursive=True  # Search subdirectories
)

# Or specify specific GPU IDs
run_inference(
    input_path='./images',
    output_path='./output',
    gpus=[2, 3],  # Use GPU 2 and GPU 3 specifically
    batch_size=16,
    compile_model=True
)
```

Advanced Configuration

```python
from fingernet.api import run_inference

run_inference(
    input_path='./images',
    output_path='./output',
    weights_path='./models/custom_weights.pth',  # Custom model weights
    gpus=[0, 1, 2, 3],  # Use 4 specific GPUs
    batch_size=32,  # Large batch per GPU
    num_workers=8,  # More workers for faster data loading
    recursive=True,  # Search all subdirectories
    mnt_degrees=True,  # Output minutiae angles in degrees (not radians)
    compile_model=True,  # Enable torch.compile optimization
    max_image_dim=1000,  # Resize large images
    strategy='full_gpu',  # Processing strategy
    num_cpu_workers=4  # CPU workers for saving results
)
```

### Low-Level API: Direct Model Usage

For more control over the inference pipeline, you can use the model directly:

```python
from fingernet.wrapper import get_fingernet
import numpy as np
from PIL import Image

# Load the model
model = get_fingernet(
    weights_path='./models/released_version/Model.pth',
    device='cuda:0'  # or 'cpu'
)

# Model is ready for inference
# model.eval() is already called internally
```

Processing WSQ Fingerprint Images

```python
import wsq
from PIL import Image
import numpy as np
from fingernet.wrapper import get_fingernet

# Load WSQ image using the wsq library
# WSQ files can be opened directly with PIL after importing wsq
import wsq
img = Image.open('fingerprint.wsq')

# Convert to numpy array (grayscale)
img_array = np.array(img, dtype=np.float32)

# Load model
model = get_fingernet(device='cuda:0')

# Prepare input: convert numpy array to torch tensor
# prepare_input handles shape conversions automatically
input_tensor = model.prepare_input(img_array)

# Run inference
outputs = model(input_tensor, minutiae_threshold=0.5)

# Access results
minutiae = outputs['minutiae']  # (N, 4) tensor: x, y, angle, quality
enhanced = outputs['enhanced_image']  # Enhanced fingerprint
mask = outputs['segmentation_mask']  # Foreground mask
orientation = outputs['orientation_field']  # Orientation field
```

Processing Standard Image Formats

```python
from PIL import Image
import numpy as np
from fingernet.wrapper import get_fingernet

# Load standard image formats (PNG, JPG, BMP, etc.)
img = Image.open('fingerprint.png').convert('L')  # Convert to grayscale
img_array = np.array(img, dtype=np.float32)

# Load model
model = get_fingernet(device='cuda:0')

# Prepare and process
input_tensor = model.prepare_input(img_array)
outputs = model(input_tensor, minutiae_threshold=0.5)

# Extract minutiae as numpy array
minutiae_np = outputs['minutiae'].cpu().numpy()
print(f"Found {len(minutiae_np)} minutiae")

# Each minutia: [x, y, orientation_radians, quality_score]
for x, y, angle, quality in minutiae_np:
    print(f"Position: ({x:.1f}, {y:.1f}), Angle: {np.rad2deg(angle):.1f}°, Quality: {quality:.3f}")
```

Batch Processing with Direct Model Access

```python
import numpy as np
from fingernet.wrapper import get_fingernet
from PIL import Image

# Load multiple images
image_paths = ['img1.png', 'img2.png', 'img3.png']
images = [np.array(Image.open(p).convert('L'), dtype=np.float32) for p in image_paths]

# Stack into batch (all images must have same dimensions)
batch = np.stack(images)  # Shape: (B, H, W)

# Load model
model = get_fingernet(device='cuda:0')

# Process batch
input_tensor = model.prepare_input(batch)
outputs = model(input_tensor)

# outputs['minutiae'] will be a list of tensors, one per image
for i, mnt in enumerate(outputs['minutiae']):
    print(f"Image {i}: {len(mnt)} minutiae detected")
```

## Command-Line Interface (CLI)

Basic Usage

```bash
fingernet infer ./images/ ./output/
```

Single GPU Inference

```bash
fingernet infer --gpus 1 --batch-size 8 ./images/ ./output/
```

Multi-GPU Inference

```bash
# Use first 2 GPUs (GPU 0 and 1)
fingernet infer --gpus 2 --batch-size 16 --compile ./images/ ./output/

# Use specific GPUs (e.g., GPU 2 and 3)
fingernet infer --gpus "[2, 3]" --batch-size 16 --compile ./images/ ./output/
```

Advanced Options

```bash
fingernet infer \
    --gpus "[0, 1, 2, 3]" \
    --batch-size 32 \
    --cores 8 \
    --compile \
    --recursive \
    --degrees \
    --max-dim 1000 \
    --strategy full_gpu \
    --cpu-workers 4 \
    ./images/ \
    ./output/
```

CLI Options Reference

- `--gpus`: GPU configuration
  - `0` or `none`: Use CPU
  - `1`: Single GPU (GPU 0)
  - `N` (N>1): Use N GPUs (0, 1, ..., N-1)
  - `"[i, j, k]"`: Use specific GPU IDs (must be quoted)
- `--batch-size` / `-b`: Batch size per GPU (default: 8)
- `--cores`: DataLoader worker threads per GPU (default: 4)
- `--compile`: Enable torch.compile for faster inference
- `--recursive`: Search input directory recursively
- `--degrees`: Output minutiae angles in degrees instead of radians
- `--max-dim`: Maximum image dimension (resize if larger)
- `--strategy`: Processing strategy (`full_gpu`, `cpu_heavy`, etc.)
- `--cpu-workers`: Number of CPU threads for saving results (default: 4)
- `--weights`: Custom model weights path

Visualization

After inference, visualize results:

```bash
fingernet plot ./output/ image_name.png --save ./visualization.png
```

## Output Format

FingerNet generates organized outputs in four separate directories:

### File Structure

```
output/
├── minutiae/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── enhanced/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── mask/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── ori/
    ├── image1.png
    ├── image2.png
    └── ...
```

### Output Descriptions

#### Minutiae Files (`minutiae/*.txt`)

CSV format with header, each line represents one minutia point:
```
x, y, angle, score
123, 456, 1.234567, 0.876543
```
- `x, y`: Pixel coordinates of the minutia
- `angle`: Orientation in radians (or degrees if `--degrees` flag is used). The angle is measured clockwise from the horizontal axis.
- `score`: Quality/confidence score [0-1]

#### Enhanced Images (`enhanced/*.png`)

Gabor-enhanced fingerprint images with improved ridge-valley contrast. Useful for visualization and further processing.

#### Segmentation Masks (`mask/*.png`)

Binary masks indicating foreground (fingerprint region) vs background. White pixels (255) represent valid fingerprint area, black pixels (0) represent background.

#### Orientation Fields (`ori/*.png`)

Ridge orientation at each pixel, encoded as grayscale images. The orientation angle (in degrees) is shifted by +90° and stored as uint8 values for visualization.

## Citation

If you use FingerNet in your research, please cite the original paper:

```bibtex
@inproceedings{tang2017fingernet,
  title={FingerNet: An unified deep network for fingerprint minutiae extraction},
  author={Tang, Yao and Gao, Fei and Feng, Jufu and Liu, Yuhang},
  booktitle={2017 IEEE International Joint Conference on Biometrics (IJCB)},
  pages={108--116},
  year={2017},
  organization={IEEE}
}
```

## Credits

**Original Authors**: Yao Tang, Fei Gao, Jufu Feng, Yuhang Liu

**PyTorch Implementation & Extensions**: João Contreras (joao.contreras@griaule.com)

This extended implementation includes significant enhancements for production-scale deployment, including multi-GPU distributed inference, optimized batch processing, and a comprehensive API for integration into larger systems.

## License

Please refer to the original FingerNet repository for licensing information regarding the model architecture and weights.

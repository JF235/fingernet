import logging
import os
import glob
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import timedelta
import warnings
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

from .wrapper import FingerNetWrapper, get_fingernet, postprocess
from .fnet_utils import get_fingernet_logger, FnetTimer, DEFAULT_WEIGHTS_PATH

logger = get_fingernet_logger('fingernet.api', level=logging.DEBUG)

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
# Note: torch.backends.cudnn.benchmark is intentionally left OFF (default).
# Enabling it autotunes a per-shape best conv algorithm and gives ~17% extra
# throughput on fixed-size workloads (e.g. BN48k 512x512), but the chosen
# algorithm differs from cuDNN's heuristic default — combined with TF32's
# non-associative matmul this flips a small fraction of outputs near the
# threshold/NMS boundary vs prior extractions. Set it explicitly in user
# code if you prefer speed over byte-level reproducibility:
#     torch.backends.cudnn.benchmark = True

class FingerprintDataset(Dataset):
    """Dataset for loading fingerprint images."""

    def __init__(self, image_paths: list[str], max_dim: int):
        self.image_paths = image_paths
        self.max_dim = max_dim

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img_pil = Image.open(img_path).convert("L")

            if img_pil.height > self.max_dim or img_pil.width > self.max_dim:
                logger.warning(
                    f"Image {os.path.basename(img_path)} with size {img_pil.size} exceeds max_dim of {self.max_dim}. "
                    "Resizing it down."
                )
                img_pil.thumbnail(
                    (self.max_dim, self.max_dim), Image.Resampling.LANCZOS
                )

            img_np = np.array(img_pil, dtype=np.float32) / 255.0

            return {
                "image": torch.from_numpy(img_np).unsqueeze(0),
                "path": img_path,
                "original_shape": img_np.shape,
            }
        except Exception as e:
            logger.warning(f"Could not load image {img_path}. Skipping. Error: {e}")
            return None

def find_image_paths(input_path: str, recursive: bool = True) -> list[str]:
    """
    Find all image paths from input.

    Args:
        input_path: Path to file, directory, or text list
        recursive: Whether to search recursively in directories

    Returns:
        List of image paths
    """
    image_paths = []

    if os.path.isfile(input_path):
        # Check if it's a text file (list of paths)
        _, ext = os.path.splitext(input_path)
        if ext.lower() in [".txt", ".list"]:
            with open(input_path, "r") as f:
                for line in f:
                    path = line.strip()
                    if path:
                        image_paths.append(path)
        else:
            # If a single file path was provided
            single_supported = [".png", ".wsq", ".bmp"]
            if ext.lower() in single_supported:
                image_paths.append(input_path)
            else:
                raise ValueError(
                    f"Only {single_supported} files are supported for inference (received: {input_path})"
                )

    elif os.path.isdir(input_path):
        # Lock: only search for PNG in directories
        extensions = ["png", "bmp"]
        for ext in extensions:
            pattern = (
                f"{input_path}/**/*.{ext}" if recursive else f"{input_path}/*.{ext}"
            )
            image_paths.extend(glob.glob(pattern, recursive=recursive))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    if not image_paths:
        raise ValueError(f"No images found in: {input_path}")

    return sorted(image_paths)

def dynamic_padding_collate(batch):
    """
    Custom collate_fn that pads images to the max size within a batch.
    It also filters out None items that may result from loading errors.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None

    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    images, paths, orig_shapes = [], [], []
    for item in batch:
        img = item["image"]
        _, h, w = img.shape
        padding = (0, max_w - w, 0, max_h - h)  # (left, right, top, bottom)
        padded_img = torch.nn.functional.pad(img, padding, mode="constant", value=1.0)

        images.append(padded_img)
        paths.append(item["path"])
        orig_shapes.append(item["original_shape"])

    batch_tensors = torch.stack(images)
    batch_paths = paths
    batch_orig_shapes = (
        torch.tensor([s[0] for s in orig_shapes]),
        torch.tensor([s[1] for s in orig_shapes]),
    )

    return batch_tensors, batch_paths, batch_orig_shapes


def save_results(result_item: dict, output_path: str, mnt_degrees: bool = True, input_base_path: str = None):
    """
    Save inference results to disk in organized structure.

    Args:
        result_item: Dictionary with keys 'input_path', 'minutiae', 'enhanced_image', etc.
        output_path: Base output directory
        mnt_degrees: If True, save minutiae angles in degrees instead of radians
        input_base_path: Base path of the input directory to preserve structure
    """
    input_path = result_item["input_path"]
    
    # Calculate relative path from input base if provided
    if input_base_path and os.path.isdir(input_base_path):
        # Get relative path from input base to the file
        rel_path = os.path.relpath(input_path, input_base_path)
        rel_dir = os.path.dirname(rel_path)
        original_filename = os.path.basename(rel_path)
    else:
        # Fallback to just using the filename (backward compatibility)
        rel_dir = ""
        original_filename = os.path.basename(input_path)
    
    base_name = os.path.splitext(original_filename)[0]

    # Save minutiae (.min) — format: CCW angle in degrees (int), quality 0-100 (int)
    minutiae = result_item["minutiae"].copy()
    angle_ccw_deg = np.round((-np.rad2deg(minutiae[:, 2])) % 360).astype(int)
    quality_int = np.round(minutiae[:, 3] * 100).astype(int)
    minutiae_out = np.column_stack([
        minutiae[:, 0].astype(int),
        minutiae[:, 1].astype(int),
        angle_ccw_deg,
        quality_int,
    ])

    minutiae_path = os.path.join(output_path, "minutiae", rel_dir, f"{base_name}.min")
    os.makedirs(os.path.dirname(minutiae_path), exist_ok=True)
    np.savetxt(
        minutiae_path,
        minutiae_out,
        fmt="%d",
        header="X Y ANGLE QUALITY",
        comments="#MIN ",
        delimiter=" ",
    )

    # Save enhanced image (.png)
    enhanced_path = os.path.join(output_path, "enhanced", rel_dir, original_filename)
    os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
    Image.fromarray(result_item["enhanced_image"]).save(enhanced_path)

    # Save mask (.png)
    mask_path = os.path.join(output_path, "mask", rel_dir, original_filename)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    Image.fromarray(result_item["segmentation_mask"]).save(mask_path)

    # Save orientation field (encoded as PNG)
    ori_cpu = result_item["orientation_field"]
    orientation_path = os.path.join(output_path, "ori", rel_dir, original_filename)
    os.makedirs(os.path.dirname(orientation_path), exist_ok=True)
    angles_deg_shifted = np.round(np.rad2deg(ori_cpu) + 90).astype(np.uint8)
    Image.fromarray(angles_deg_shifted).save(orientation_path)

    # Continuous quality mask (sigmoid output, per-pixel confidence)
    quality_path = os.path.join(output_path, "quality", rel_dir, original_filename)
    os.makedirs(os.path.dirname(quality_path), exist_ok=True)
    Image.fromarray(result_item["quality"]).save(quality_path)

    # ----- Opt-in outputs (--full) ----- #

    # Modulated enhanced/orientation (multiplied by the mask) — --full only
    if 'enhanced_image_mod' in result_item:
        path = os.path.join(output_path, "enhanced_mod", rel_dir, original_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(result_item["enhanced_image_mod"]).save(path)

    if 'orientation_field_mod' in result_item:
        path = os.path.join(output_path, "ori_mod", rel_dir, original_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        angles = np.round(np.rad2deg(result_item["orientation_field_mod"]) + 90).astype(np.uint8)
        Image.fromarray(angles).save(path)

def assemble_result(final_outputs: dict, i: int, orig_h: int, orig_w: int,
                    input_path: str, full: bool = False) -> dict:
    """One image's products, cropped back to its original size.

    `orientation_field_mod` is derived here because it is exactly the mask times a
    primitive; the other two `--full` products come from `postprocess`, which is the
    only place that still has the float enhanced image and the unmasked score (see
    its docstring for why neither is derivable).
    """
    mask = final_outputs['segmentation_mask'][i]
    crop = lambda t: t[:orig_h, :orig_w].cpu().numpy()
    result = {
        'input_path': input_path,
        'minutiae': final_outputs['minutiae'][i].cpu().numpy(),
        'enhanced_image': crop(final_outputs['enhanced_image'][i]),
        'segmentation_mask': crop(mask),
        'orientation_field': crop(final_outputs['orientation_field'][i]),
        'quality': crop(final_outputs['quality'][i]),
    }
    if full:
        mask01 = (mask / 255.0).float()
        result['enhanced_image_mod'] = crop(final_outputs['enhanced_image_mod'][i])
        # the one product that IS the mask times a primitive
        result['orientation_field_mod'] = crop(final_outputs['orientation_field'][i] * mask01)
    return result


def postprocess_and_save_batch(
    raw_outputs_cpu: dict,
    batch_paths: list[str],
    batch_orig_shapes: tuple,
    padded_shape: tuple,
    output_path: str,
    mnt_degrees: bool,
    input_base_path: str = None,
    full: bool = False,
    threshold: float = 0.05,
):
    """Run post-processing and save the results of one batch."""
    worker_id = threading.get_ident()

    logger.info("CPU worker started processing batch", extra={'cpu_worker_id': worker_id, 'first_image': os.path.basename(batch_paths[0])})
    try:
        with FnetTimer("Post-processing", logger) as t_post:
            final_outputs = postprocess(raw_outputs_cpu, threshold=threshold)

        padded_h, padded_w = padded_shape

        for i in range(len(batch_paths)):
            orig_h, orig_w = batch_orig_shapes[0][i].item(), batch_orig_shapes[1][i].item()
            result_item = assemble_result(final_outputs, i, orig_h, orig_w,
                                          batch_paths[i], full=full)
            save_results(result_item, output_path, mnt_degrees, input_base_path)

        
        logger.info(
            "CPU worker finished batch", 
            extra={
                'cpu_worker_id': worker_id, 
                'batch_size': len(batch_paths)
            }
        )
    except Exception as e:
        warnings.warn(f"Post-processing failed for the batch starting at {os.path.basename(batch_paths[0])}. Error: {e}")

def create_output_directories(output_path: str, full: bool = False):
    """Create the output directory structure.

    Always created: ``minutiae/``, ``mask/``, ``enhanced/``, ``ori/``,
    ``quality/``. ``ori/`` holds the orientation field WITHOUT mask
    modulation and ``enhanced/`` holds the enhanced image WITHOUT mask
    modulation; the "raw" versions preserve full spatial information.
    ``quality/`` holds the continuous sigmoid mask (per-pixel
    confidence).

    Optional (``full=True``):
        Also creates ``enhanced_mod/`` and ``ori_mod/`` (mask-modulated).
    """
    os.makedirs(os.path.join(output_path, "minutiae"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "mask"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "ori"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "quality"), exist_ok=True)
    if full:
        os.makedirs(os.path.join(output_path, "enhanced_mod"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "ori_mod"), exist_ok=True)

def setup_ddp(rank: int, world_size: int, local_device_idx: int, timeout_minutes: int = 30):
    """
    Initialize distributed process group.

    Args:
        rank: Unique identifier for this process (0, 1, ...)
        world_size: Total number of processes
        local_device_idx: Logical CUDA device index for this process (0..world_size-1)
        timeout_minutes: Timeout for DDP operations
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set device for this process BEFORE init_process_group.
    # CUDA_VISIBLE_DEVICES remaps selected physical GPUs to logical indices.
    torch.cuda.set_device(local_device_idx)

    # Initialize process group with proper timeout and device_id
    dist.init_process_group(
        backend="nccl",  # nccl stands for NVIDIA Collective Communications Library
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
        device_id=torch.device(f"cuda:{local_device_idx}"),
    )

def cleanup_ddp():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def _ddp_launch_target(rank: int, world_size: int, config: dict):
    """Target function for mp.spawn."""
    # With CUDA_VISIBLE_DEVICES already applied, rank == logical GPU index.
    runner = InferenceRunner(config)
    runner.setup(rank, world_size, rank)
    runner.run()

def run_inference(
    input_path: str,
    output_path: str,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    gpus: int | list[int] | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    recursive: bool = True,
    mnt_degrees: bool = True,
    threshold: float = 0.05,
    compile_model: bool = False,
    max_image_dim: int = 1024,
    strategy: str = 'full_gpu',
    num_cpu_workers: int = 4,
    full: bool = False,
):
    """
    Run FingerNet inference on images.

    Args:
        input_path: Path to image, directory, or text file with image paths
        output_path: Directory to save results
        weights_path: Path to model weights (.pth file)
        gpus: GPU configuration:
            - None or 0: Use CPU
            - int (e.g., 1): Use first N GPUs (logical IDs 0..N-1 after remap)
            - int (e.g., 2): Use 2 GPUs with DDP (logical IDs 0,1)
            - list[int] (e.g., [2,3]): Select specific physical GPUs in CLI, then use logical 0,1
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers per GPU
        recursive: Search for images recursively
        mnt_degrees: Save minutiae angles in degrees instead of radians
        compile_model: Use torch.compile for faster inference
        full: also export ``enhanced_mod/`` and ``ori_mod/`` (the mask-modulated
            counterparts).

    Example:
        >>> run_inference('images/', 'output/', gpus=2, batch_size=8)
    """
    image_paths = find_image_paths(input_path, recursive)

    # Base path for preserving directory structure under output_path.
    #
    # For a directory, that is the directory. For a .txt list it is NOT the list's own
    # directory: the list can live anywhere, and a base the images are not under makes
    # save_results' relpath emit "../.." — every product then lands OUTSIDE output_path,
    # and because the five per-image products differ only by their leading directory,
    # they collapse onto one filename and overwrite each other. Silent 4/5 data loss.
    # The common ancestor of the images themselves is the base that cannot do that.
    if os.path.isdir(input_path):
        input_base_path = input_path
    else:
        input_base_path = os.path.commonpath([os.path.dirname(p) for p in image_paths])

    # Collect all config into one dict for convenience
    config = locals()

    use_cpu = (gpus is None or gpus == 0 or not torch.cuda.is_available())
    requested_world_size = 0
    if not use_cpu:
        if isinstance(gpus, int):
            requested_world_size = gpus
        elif isinstance(gpus, list):
            requested_world_size = len(gpus)
        else:
            raise ValueError(f"Unsupported GPU configuration type: {type(gpus)}")

    if not use_cpu:
        visible_cuda_count = torch.cuda.device_count()
        if requested_world_size > visible_cuda_count:
            visible_names = []
            for i in range(visible_cuda_count):
                try:
                    visible_names.append(f"cuda:{i}={torch.cuda.get_device_name(i)}")
                except Exception:
                    visible_names.append(f"cuda:{i}=<unavailable>")
            visible_desc = ", ".join(visible_names) if visible_names else "<none>"
            raise ValueError(
                f"Requested {requested_world_size} GPU(s), but only {visible_cuda_count} "
                f"visible to CUDA. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} | "
                f"visible devices: {visible_desc}. "
                "This usually means one or more requested GPU IDs are invalid for this host, "
                "or a parent environment/scheduler already restricted visible GPUs."
            )
    is_ddp = (not use_cpu and requested_world_size > 1)

    if use_cpu:
        logger.info("Starting Inference on CPU")
        runner = InferenceRunner(config)
        runner.setup()
        runner.run()

    elif is_ddp:
        world_size = requested_world_size
        logger.info(
            f"Starting Distributed Inference on {world_size} logical GPU(s). "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

        mp.spawn(
            _ddp_launch_target,
            nprocs=world_size,
            args=(world_size, config),
            join=True
        )
    else: # Single GPU
        logger.info(
            "Starting Inference on single logical GPU (cuda:0). "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
        config['gpus'] = True
        runner = InferenceRunner(config)
        runner.setup(rank=-1, world_size=1, gpu_id=0)
        runner.run()


def _save_results_chunk(
    results_chunk: list[dict],
    output_path: str,
    mnt_degrees: bool,
    worker_rank: int = -1,
    input_base_path: str = None
):
    """Worker thread target. Receives a chunk of results and saves them to disk."""
    desc = f"Saving (Worker {worker_rank})" if worker_rank >= 0 else "Saving Results"

    # Workers shouldn't show individual progress bars (many can run in
    # parallel). Just iterate and save.
    for result_item in results_chunk:
        try:
            save_results(result_item, output_path, mnt_degrees, input_base_path)
        except Exception as e:
            # Logging matters in threads — otherwise failures are silent.
            base_name = os.path.basename(result_item.get('input_path', 'unknown_file'))
            logger.warning(f"Failed to save result for {base_name} in chunk. Error: {e}")


class InferenceRunner:
    def __init__(self, config: dict):
        """Initialize the runner with all required configuration.

        ``config`` is a dict that holds everything: image_paths,
        output_path, batch_size, etc.
        """
        self.config = config
        self.strategy = None
        self.rank = -1
        self.world_size = 1
        self.device = "cpu"
        self.is_main_process = True
        self.model = None
        self.dataloader = None

    def setup(self, rank: int = -1, world_size: int = 1, gpu_id: int = 0):
        """Configure the environment for this worker process.

        Sets up DDP, device, model, and dataloader.

        Args:
            rank: DDP process rank (-1 for single-process mode)
            world_size: Total number of processes
            gpu_id: Logical GPU index to use (for DDP)
        """
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank <= 0) # Rank 0 para DDP, -1 para single/cpu

        # 1. Configure DDP and device
        if self.world_size > 1:
            setup_ddp(self.rank, self.world_size, gpu_id)
            # Use the correct logical index after CUDA_VISIBLE_DEVICES remap
            self.device = f"cuda:{gpu_id}"
        elif self.config['gpus'] and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        if self.is_main_process:
            logger.info(f"Setting up runner on device: {self.device}")
            create_output_directories(
                self.config['output_path'],
                full=self.config.get('full', False),
            )
        
        if self.world_size > 1:
            dist.barrier()  # Ensure the output dirs are visible to all ranks

        # 2. Load the model
        self.model = get_fingernet(
            weights_path=self.config['weights_path'],
            device=self.device
        )
        if self.config['compile_model']:
            if self.is_main_process: logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # No need to wrap the model with DDP for inference — keeps things simple.

        # 3. Set up dataset and dataloader
        dataset = FingerprintDataset(self.config['image_paths'], max_dim=self.config['max_image_dim'])
        sampler = None
        shuffle = False
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False
            )
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=(self.config['num_workers'] > 0),
            collate_fn=dynamic_padding_collate,
        )

        # 4. Pick execution strategy
        self.strategy = self.config['strategy']

    def run(self):
        """Dispatch to the right execution method based on the strategy."""
        if self.is_main_process:
            logger.info(f"Executing with strategy: '{self.strategy}'")

        if self.strategy == 'hybrid':
            self._run_hybrid()
        elif self.strategy == 'full_gpu':
            self._run_full_gpu()
        else:
            raise ValueError(f"Unknown execution strategy: {self.strategy}")

        if self.world_size > 1:
            dist.barrier()
            cleanup_ddp()

        if self.is_main_process:
            logger.info("✓ Inference Complete!")
    
    def _run_hybrid(self):
        """Throughput-oriented pipeline (GPU forward, CPU post-processing)."""
        # Uses ThreadPoolExecutor to overlap CPU post-proc with the next GPU batch.
        if self.is_main_process:
            logger.info("Starting inference loop...")

        num_cpu_workers = self.config['num_cpu_workers']
        
        with ThreadPoolExecutor(max_workers=num_cpu_workers) as executor:
            futures = []
            max_queue_size = 2 * num_cpu_workers

            with torch.inference_mode():
                desc = f"GPU {self.rank}" if self.world_size > 1 else "Processing"
                iterator = tqdm(self.dataloader, desc=desc, disable=not self.is_main_process)

                for batch_tensors, batch_paths, batch_orig_shapes in iterator:
                    if batch_tensors is None: continue

                    # --- GPU INFERENCE STAGE ---
                    _, _, padded_h, padded_w = batch_tensors.shape
                    batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                    raw_outputs = self.model.fingernet(batch_tensors)  # core model

                    # --- D2H TRANSFER ---
                    raw_outputs_cpu = {k: v.detach().cpu() for k, v in raw_outputs.items()}

                    # --- DISPATCH TO CPU WORKERS ---
                    future = executor.submit(
                        postprocess_and_save_batch,
                        raw_outputs_cpu, batch_paths, batch_orig_shapes,
                        (padded_h, padded_w), self.config['output_path'], self.config['mnt_degrees'],
                        self.config.get('input_base_path'),
                        self.config.get('full', False),
                        self.config.get('threshold', 0.05),
                    )
                    futures.append(future)

                    # Bound the queue to avoid runaway memory usage
                    if len(futures) >= max_queue_size:
                        futures.pop(0).result()  # wait for the oldest to drain

            # Wait for everything still in flight
            if self.is_main_process:
                logger.info("Inference complete. Finalizing post-processing...")
            for future in tqdm(futures, desc=f"Finalizing (Worker {self.rank})", disable=not self.is_main_process):
                future.result()

        
    def _run_full_gpu(self):
        """Latency-oriented pipeline (everything on the GPU), with async save."""
        # Same CPU-worker count as the hybrid strategy, for consistency
        num_save_workers = self.config['num_cpu_workers']

        # Chunk that triggers a save dispatch (e.g. batch_size 8 × 10 = 80 imgs).
        # Avoids dispatching too-small tasks to the save workers.
        chunk_size = self.config['batch_size'] * 10

        # Background pool that saves results to disk
        with ThreadPoolExecutor(max_workers=num_save_workers) as save_executor:
            futures = []
            save_chunk = []

            with torch.inference_mode():
                desc = f"GPU {self.rank}" if self.world_size > 1 else "Processing"
                iterator = tqdm(self.dataloader, desc=desc, disable=not self.is_main_process)

                for batch_tensors, batch_paths, batch_orig_shapes in iterator:
                    if batch_tensors is None: continue

                    # --- GPU STAGE: forward + post-processing ---
                    batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                    _full = self.config.get('full', False)
                    final_outputs = self.model(
                        batch_tensors,
                        minutiae_threshold=self.config.get('threshold', 0.05),
                    )

                    # --- D2H + COLLECT (cheap) ---
                    for i in range(len(batch_paths)):
                        orig_h, orig_w = batch_orig_shapes[0][i].item(), batch_orig_shapes[1][i].item()
                        save_chunk.append(assemble_result(
                            final_outputs, i, orig_h, orig_w, batch_paths[i], full=_full))

                    # --- DISPATCH SAVE CHUNK ---
                    if len(save_chunk) >= chunk_size:
                        future = save_executor.submit(
                            _save_results_chunk,
                            save_chunk,
                            self.config['output_path'],
                            self.config['mnt_degrees'],
                            self.rank,
                            self.config.get('input_base_path')
                        )
                        futures.append(future)
                        save_chunk = []  # reset for the next cycle

            # --- FINALIZATION ---
            # Dispatch the last (possibly smaller) chunk
            if save_chunk:
                future = save_executor.submit(
                    _save_results_chunk,
                    save_chunk,
                    self.config['output_path'],
                    self.config['mnt_degrees'],
                    self.rank,
                    self.config.get('input_base_path')
                )
                futures.append(future)

            # Wait for all save workers to drain
            if self.is_main_process:
                logger.info("Inference complete. Waiting for save workers to finish...")

            # Use a progress bar to give feedback while waiting on the futures
            for future in tqdm(futures, desc=f"Finalizing Save (Worker {self.rank})", disable=not self.is_main_process):
                future.result()  # waits for completion and re-raises exceptions

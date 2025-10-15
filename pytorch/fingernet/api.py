import logging
import os
import glob
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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

torch.set_float32_matmul_precision("medium")

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

def find_image_paths(input_path: str, recursive: bool = False) -> list[str]:
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


def save_results(result_item: dict, output_path: str, mnt_degrees: bool = False):
    """
    Save inference results to disk in organized structure.

    Args:
        result_item: Dictionary with keys 'input_path', 'minutiae', 'enhanced_image', etc.
        output_path: Base output directory
        mnt_degrees: If True, save minutiae angles in degrees instead of radians
    """
    original_filename = os.path.basename(result_item["input_path"])
    base_name = os.path.splitext(original_filename)[0]

    # Save minutiae (.txt)
    minutiae = result_item["minutiae"].copy()
    if mnt_degrees:
        minutiae[:, 2] = np.round(np.rad2deg(minutiae[:, 2]), 2)

    minutiae_path = os.path.join(output_path, "minutiae", f"{base_name}.txt")
    np.savetxt(
        minutiae_path,
        minutiae,
        fmt=["%.0f", "%.0f", "%.6f", "%.6f"],
        header="x, y, angle, score",
        delimiter=",",
    )

    # Save enhanced image (.png)
    enhanced_path = os.path.join(output_path, "enhanced", original_filename)
    Image.fromarray(result_item["enhanced_image"]).save(enhanced_path)

    # Save mask (.png)
    mask_path = os.path.join(output_path, "mask", original_filename)
    Image.fromarray(result_item["segmentation_mask"]).save(mask_path)

    # Save orientation field (encoded as PNG)
    ori_cpu = result_item["orientation_field"]
    orientation_path = os.path.join(output_path, "ori", original_filename)
    angles_deg_shifted = np.round(np.rad2deg(ori_cpu) + 90).astype(np.uint8)
    Image.fromarray(angles_deg_shifted).save(orientation_path)

def postprocess_and_save_batch(
    raw_outputs_cpu: dict,
    batch_paths: list[str],
    batch_orig_shapes: tuple,
    padded_shape: tuple,
    output_path: str,
    mnt_degrees: bool
):
    """Executa pós-processamento e salva os resultados de um lote."""
    worker_id = threading.get_ident()

    logger.info("CPU worker started processing batch", extra={'cpu_worker_id': worker_id, 'first_image': os.path.basename(batch_paths[0])})
    try:
        with FnetTimer("Post-processing", logger) as t_post:
            final_outputs = postprocess(raw_outputs_cpu, threshold=0.5)

        padded_h, padded_w = padded_shape

        for i in range(len(batch_paths)):
            orig_h, orig_w = batch_orig_shapes[0][i].item(), batch_orig_shapes[1][i].item()

            minutiae = final_outputs["minutiae"][i].numpy()

            # Correção de coordenadas devido ao padding dinâmico
            # Esta lógica precisa ser ajustada, pois o padding era centralizado.
            # Por simplicidade, assumimos padding à direita/inferior como no código original.
            enhanced_img = final_outputs["enhanced_image"][i][:orig_h, :orig_w].numpy()
            seg_mask = final_outputs["segmentation_mask"][i][:orig_h, :orig_w].numpy()
            ori_field = final_outputs["orientation_field"][i][:orig_h, :orig_w].numpy()

            result_item = {
                "input_path": batch_paths[i],
                "minutiae": minutiae,
                "enhanced_image": enhanced_img,
                "segmentation_mask": seg_mask,
                "orientation_field": ori_field,
            }
            save_results(result_item, output_path, mnt_degrees)

        
        logger.info(
            "CPU worker finished batch", 
            extra={
                'cpu_worker_id': worker_id, 
                'batch_size': len(batch_paths)
            }
        )
    except Exception as e:
        warnings.warn(f"Falha no pós-processamento do lote iniciado com {os.path.basename(batch_paths[0])}. Erro: {e}")

def create_output_directories(output_path: str):
    """Create output directory structure."""
    os.makedirs(os.path.join(output_path, "minutiae"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "mask"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "ori"), exist_ok=True)


def setup_ddp(rank: int, world_size: int, timeout_minutes: int = 30):
    """
    Initialize distributed process group.

    Args:
        rank: Unique identifier for this process
        world_size: Total number of processes
        timeout_minutes: Timeout for DDP operations
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set device for this process BEFORE init_process_group
    torch.cuda.set_device(rank)

    # Initialize process group with proper timeout and device_id
    dist.init_process_group(
        backend="nccl",  # nccl stands for NVIDIA Collective Communications Library
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
        device_id=torch.device(f"cuda:{rank}"),
    )


def cleanup_ddp():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def inference_worker_ddp_hybrid(
    rank: int,
    world_size: int,
    image_paths: list[str],
    max_image_dim: int,
    output_path: str,
    weights_path: str,
    batch_size: int,
    num_workers: int,
    mnt_degrees: bool,
    compile_model: bool,
    num_cpu_workers: int,
):
    """
    Worker function for distributed inference on a single GPU.
    """
    try:
        # Setup DDP
        setup_ddp(rank, world_size)

        # Only rank 0 prints and creates directories
        is_main = (rank == 0)

        if is_main:
            logger.info(f"Starting Distributed Inference")
            logger.info(f"World Size: {world_size} GPUs")
            logger.info(f"Batch Size per GPU: {batch_size}")
            logger.info(f"Total Images: {len(image_paths)}")
            logger.info(f"Output Path: {output_path}")

            # Create output directories
            create_output_directories(output_path)

        # Synchronize: all processes wait for rank 0 to create directories
        dist.barrier()

        # Load model
        if is_main:
            logger.info(f"[Rank {rank}] Loading model...")

        model: FingerNetWrapper = get_fingernet(
            weights_path=weights_path, device=f"cuda:{rank}"
        )
        model.eval()

        # Optionally compile model
        if compile_model:
            if is_main:
                logger.info(f"[Rank {rank}] Compiling model with torch.compile...")
            model = torch.compile(model)

        # Wrap with DDP - not strictly necessary for inference-only but helps with synchronization
        # model = DDP(model, device_ids=[rank])

        # Setup dataset and distributed sampler
        dataset = FingerprintDataset(image_paths, max_dim=max_image_dim)
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=dynamic_padding_collate,
        )

        if is_main:
            logger.info(f"[Rank {rank}] Starting inference...")
        
        with ThreadPoolExecutor(max_workers=num_cpu_workers) as executor:
            futures = []
            max_queue_size = 2 * num_cpu_workers
            with torch.no_grad():
                iterator = tqdm(dataloader, desc=f"GPU {rank}", disable=not is_main)
                for batch_tensors, batch_paths, batch_orig_shapes in iterator:
                    if batch_tensors is None: continue

                    _, _, padded_h, padded_w = batch_tensors.shape
                    batch_tensors = batch_tensors.to(f"cuda:{rank}")
                    # ETAPA GPU: Inferência
                    with FnetTimer('GPU Inference', logger) as t_gpu:
                        raw_outputs = model.time(batch_tensors)


                    # ETAPA DE TRANSFERÊNCIA: Mover para CPU
                    with FnetTimer('GPU->CPU Transfer', logger) as t_transfer:
                        raw_outputs_cpu = {k: v.detach().cpu() for k, v in raw_outputs.items()}

                    # ETAPA DE SUBMISSÃO: Enviar para a pool de threads
                    future = executor.submit(
                        postprocess_and_save_batch,
                        raw_outputs_cpu,
                        batch_paths,
                        batch_orig_shapes,
                        (padded_h, padded_w),
                        output_path,
                        mnt_degrees
                    )
                    futures.append(future)
                    queue_size = len(futures)
                    logger.info(
                        "Task submitted to CPU queue",
                        extra={'queue_size': queue_size, 'first_image': os.path.basename(batch_paths[0])}
                    )

                    if queue_size >= max_queue_size:
                        logger.warning("CPU queue is full. GPU process is waiting.", extra={'queue_size': queue_size})
                        with FnetTimer('GPU waiting', logger) as t_wait:
                            completed_future = futures.pop(0)
                            completed_future.result()
                        logger.info("GPU process resumed.")

            # Aguardar a finalização de todas as tarefas submetidas por este rank
            if is_main:
                logger.info(f"[Rank {rank}] Inference complete. Waiting for post-processing...")
            for future in tqdm(futures, desc=f"Finalizing (GPU {rank})", disable=not is_main):
                future.result()

        # Sincronizar todas as GPUs antes de finalizar
        dist.barrier()
        if is_main:
            logger.info(f"✓ Inference Complete!")
            logger.info(f"  Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"[Rank {rank}] Error: {e}")
        raise
    finally:
        # Cleanup
        cleanup_ddp()


def inference_worker_ddp_gpu(
    rank: int,
    world_size: int,
    image_paths: list[str],
    max_image_dim: int,
    output_path: str,
    weights_path: str,
    batch_size: int,
    num_workers: int,
    mnt_degrees: bool,
    compile_model: bool,
    num_cpu_workers: int,
):
    """
    Worker function for distributed inference on a single GPU.
    """
    try:
        # Setup DDP
        setup_ddp(rank, world_size)

        # Only rank 0 prints and creates directories
        is_main = (rank == 0)

        if is_main:
            logger.info(f"Starting Distributed Inference")
            logger.info(f"World Size: {world_size} GPUs")
            logger.info(f"Batch Size per GPU: {batch_size}")
            logger.info(f"Total Images: {len(image_paths)}")
            logger.info(f"Output Path: {output_path}")

            # Create output directories
            create_output_directories(output_path)

        # Synchronize: all processes wait for rank 0 to create directories
        dist.barrier()

        # Load model
        if is_main:
            logger.info(f"[Rank {rank}] Loading model...")

        model: FingerNetWrapper = get_fingernet(
            weights_path=weights_path, device=f"cuda:{rank}"
        )
        model.eval()

        # Optionally compile model
        if compile_model:
            if is_main:
                logger.info(f"[Rank {rank}] Compiling model with torch.compile...")
            model = torch.compile(model)

        # Wrap with DDP - not strictly necessary for inference-only but helps with synchronization
        # model = DDP(model, device_ids=[rank])

        # Setup dataset and distributed sampler
        dataset = FingerprintDataset(image_paths, max_dim=max_image_dim)
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=dynamic_padding_collate,
        )

        if is_main:
            logger.info(f"[Rank {rank}] Starting inference...")
        
        all_results = []
        with torch.no_grad():
            iterator = tqdm(dataloader, desc=f"GPU {rank}", disable=not is_main)
            for batch_tensors, batch_paths, batch_orig_shapes in iterator:
                if batch_tensors is None: continue

                _, _, padded_h, padded_w = batch_tensors.shape
                batch_tensors = batch_tensors.to(f"cuda:{rank}")
                # ETAPA GPU: Inferência
                with FnetTimer('GPU Inference', logger) as t_gpu:
                    results = model(batch_tensors)


                # ETAPA DE TRANSFERÊNCIA: Mover para CPU
                with FnetTimer('GPU->CPU Transfer', logger) as t_transfer:
                    for i in range(len(batch_paths)):
                        result_item = {
                            'input_path': batch_paths[i],
                            'minutiae': results['minutiae'][i].cpu().numpy(),
                            'enhanced_image': results['enhanced_image'][i].cpu().numpy(),
                            'segmentation_mask': results['segmentation_mask'][i].cpu().numpy(),
                            'orientation_field': results['orientation_field'][i].cpu().numpy(),
                        }
                        all_results.append(result_item)


        dist.barrier()  # Sincronizar antes de salvar

        for result in tqdm(all_results, desc=f"Saving (GPU {rank})", disable = not is_main):
            save_results(result, output_path, mnt_degrees)

        # ALL processes must participate in reduce (collective operation)
        total_processed = torch.tensor(len(all_results), device=f'cuda:{rank}')
        dist.reduce(total_processed, dst=0, op=dist.ReduceOp.SUM)

        # Sincronizar todas as GPUs antes de finalizar
        dist.barrier()
        if is_main:
            logger.info(f"✓ Inference Complete!")
            logger.info(f"  Total images processed: {total_processed.item()}")
            logger.info(f"  Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"[Rank {rank}] Error: {e}")
        raise
    finally:
        # Cleanup
        cleanup_ddp()


def inference_single_gpu(
    device_id: int,
    image_paths: list[str],
    max_image_dim: int,
    output_path: str,
    weights_path: str,
    batch_size: int,
    num_workers: int,
    mnt_degrees: bool,
    compile_model: bool,
    num_cpu_workers: int,
):
    """
    Inference on a single GPU (or CPU if device_id is -1).
    """
    device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id != -1 else "cpu"

    logger.info(f"Starting Single GPU Inference")
    logger.info(f"Device: {device}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Total Images: {len(image_paths)}")
    logger.info(f"Output Path: {output_path}")

    create_output_directories(output_path)
    model = get_fingernet(weights_path=weights_path, device=device, log=False)
    if compile_model:
        model = torch.compile(model)

    dataset = FingerprintDataset(image_paths, max_dim=max_image_dim)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, shuffle=False, persistent_workers=(num_workers > 0),
        collate_fn=dynamic_padding_collate
    )

    with ThreadPoolExecutor(max_workers=num_cpu_workers) as executor:
        futures = []
        max_queue_size = 3 * num_cpu_workers
        with torch.no_grad():
            for batch_tensors, batch_paths, batch_orig_shapes in tqdm(dataloader, desc="Processing"):
                if batch_tensors is None: continue

                # ETAPA GPU: Inferência
                _, _, padded_h, padded_w = batch_tensors.shape
                batch_tensors = batch_tensors.to(device)
                raw_outputs = model(batch_tensors)

                # ETAPA DE TRANSFERÊNCIA: Mover para CPU
                raw_outputs_cpu = {k: v.detach().cpu() for k, v in raw_outputs.items()}

                # ETAPA DE SUBMISSÃO: Enviar para a pool de threads
                future = executor.submit(
                    postprocess_and_save_batch,
                    raw_outputs_cpu,
                    batch_paths,
                    batch_orig_shapes,
                    (padded_h, padded_w),
                    output_path,
                    mnt_degrees
                )
                futures.append(future)

                if len(futures) >= max_queue_size:
                    completed_future = futures.pop(0)
                    completed_future.result()

        logger.info("\nInference complete. Waiting for post-processing and saving to finish...")
        for future in tqdm(futures, desc="Finalizing"):
            future.result()  # Aguarda a conclusão e levanta exceções se houver

    logger.info(f"✓ Inference Complete!")
    logger.info(f"  Total images processed: {len(image_paths)}")
    logger.info(f"  Results saved to: {output_path}")


def run_inference(
    input_path: str,
    output_path: str,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    gpus: int | list[int] | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    recursive: bool = False,
    mnt_degrees: bool = True,
    compile_model: bool = False,
    max_image_dim: int = 1024,
    num_cpu_workers: int = 2,
):
    """
    Run FingerNet inference on images.

    Args:
        input_path: Path to image, directory, or text file with image paths
        output_path: Directory to save results
        weights_path: Path to model weights (.pth file)
        gpus: GPU configuration:
            - None or 0: Use CPU
            - int (e.g., 1): Use single GPU with ID 0
            - int (e.g., 2): Use 2 GPUs with DDP (IDs 0,1)
            - list[int] (e.g., [2,3]): Use specific GPUs with DDP
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers per GPU
        recursive: Search for images recursively
        mnt_degrees: Save minutiae angles in degrees instead of radians
        compile_model: Use torch.compile for faster inference

    Example:
        >>> run_inference('images/', 'output/', gpus=2, batch_size=8)
    """
    # Find all images
    # print("Discovering images...")
    image_paths = find_image_paths(input_path, recursive)

    if gpus is None or gpus == 0 or not torch.cuda.is_available():
        inference_single_gpu(
            device_id=-1,
            image_paths=image_paths,
            max_image_dim=max_image_dim,
            output_path=output_path,
            weights_path=weights_path,
            batch_size=batch_size,
            num_workers=num_workers,
            mnt_degrees=mnt_degrees,
            compile_model=compile_model,
            num_cpu_workers=num_cpu_workers,
        )
    elif isinstance(gpus, int) or isinstance(gpus, list):
        if isinstance(gpus, int):
            world_size = gpus
            gpu_ids = list(range(world_size))
        else:
            world_size = len(gpus)
            gpu_ids = gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        mp.spawn(
            inference_worker_ddp_gpu,
            nprocs=world_size,
            join=True,
            args=(
                world_size,
                image_paths,
                max_image_dim,
                output_path,
                weights_path,
                batch_size,
                num_workers,
                mnt_degrees,
                compile_model,
                num_cpu_workers,
            ),
        )
    else:
        raise ValueError(f"Invalid gpus parameter: {gpus}")
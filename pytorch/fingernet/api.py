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

    # Save minutiae (.min) - padrão: ângulo CCW em graus (int), qualidade 0-100 (int)
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

    # Save quality mask (.png) if present
    if 'quality_mask' in result_item:
        qmask_path = os.path.join(output_path, "quality_mask", rel_dir, original_filename)
        os.makedirs(os.path.dirname(qmask_path), exist_ok=True)
        Image.fromarray(result_item["quality_mask"]).save(qmask_path)

    # Save unmodulated enhanced image (.png) if present
    if 'enhanced_image_unmod' in result_item:
        path = os.path.join(output_path, "enhanced_unmod", rel_dir, original_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(result_item["enhanced_image_unmod"]).save(path)

    # Save unmodulated orientation field (encoded as PNG) if present
    if 'orientation_field_unmod' in result_item:
        path = os.path.join(output_path, "ori_unmod", rel_dir, original_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        angles = np.round(np.rad2deg(result_item["orientation_field_unmod"]) + 90).astype(np.uint8)
        Image.fromarray(angles).save(path)

    # Save unmodulated minutiae (.min) if present
    if 'minutiae_unmod' in result_item:
        mnt_unmod = result_item["minutiae_unmod"].copy()
        angle_ccw_deg_unmod = np.round((-np.rad2deg(mnt_unmod[:, 2])) % 360).astype(int)
        quality_int_unmod = np.round(mnt_unmod[:, 3] * 100).astype(int)
        mnt_unmod_out = np.column_stack([
            mnt_unmod[:, 0].astype(int),
            mnt_unmod[:, 1].astype(int),
            angle_ccw_deg_unmod,
            quality_int_unmod,
        ])
        mnt_unmod_path = os.path.join(output_path, "minutiae_unmod", rel_dir, f"{base_name}.min")
        os.makedirs(os.path.dirname(mnt_unmod_path), exist_ok=True)
        np.savetxt(
            mnt_unmod_path, mnt_unmod_out,
            fmt="%d",
            header="X Y ANGLE QUALITY", comments="#MIN ", delimiter=" "
        )

def postprocess_and_save_batch(
    raw_outputs_cpu: dict,
    batch_paths: list[str],
    batch_orig_shapes: tuple,
    padded_shape: tuple,
    output_path: str,
    mnt_degrees: bool,
    input_base_path: str = None,
    quality_mask: bool = False,
    unmodulated: bool = False,
    threshold: float = 0.5,
):
    """Executa pós-processamento e salva os resultados de um lote."""
    worker_id = threading.get_ident()

    logger.info("CPU worker started processing batch", extra={'cpu_worker_id': worker_id, 'first_image': os.path.basename(batch_paths[0])})
    try:
        with FnetTimer("Post-processing", logger) as t_post:
            final_outputs = postprocess(raw_outputs_cpu, threshold=threshold,
                                        quality_mask=quality_mask, unmodulated=unmodulated)

        padded_h, padded_w = padded_shape

        for i in range(len(batch_paths)):
            orig_h, orig_w = batch_orig_shapes[0][i].item(), batch_orig_shapes[1][i].item()

            minutiae = final_outputs["minutiae"][i].numpy()

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

            if quality_mask:
                result_item["quality_mask"] = final_outputs["quality_mask"][i][:orig_h, :orig_w].numpy()
            if unmodulated:
                result_item["enhanced_image_unmod"] = final_outputs["enhanced_image_unmod"][i][:orig_h, :orig_w].numpy()
                result_item["orientation_field_unmod"] = final_outputs["orientation_field_unmod"][i][:orig_h, :orig_w].numpy()
                result_item["minutiae_unmod"] = final_outputs["minutiae_unmod"][i].numpy()

            save_results(result_item, output_path, mnt_degrees, input_base_path)

        
        logger.info(
            "CPU worker finished batch", 
            extra={
                'cpu_worker_id': worker_id, 
                'batch_size': len(batch_paths)
            }
        )
    except Exception as e:
        warnings.warn(f"Falha no pós-processamento do lote iniciado com {os.path.basename(batch_paths[0])}. Erro: {e}")

def create_output_directories(output_path: str, quality_mask: bool = False, unmodulated: bool = False):
    """Create output directory structure."""
    os.makedirs(os.path.join(output_path, "minutiae"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "mask"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "ori"), exist_ok=True)
    if quality_mask:
        os.makedirs(os.path.join(output_path, "quality_mask"), exist_ok=True)
    if unmodulated:
        os.makedirs(os.path.join(output_path, "enhanced_unmod"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "ori_unmod"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "minutiae_unmod"), exist_ok=True)

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
    """Função alvo para mp.spawn."""
    # Com CUDA_VISIBLE_DEVICES já aplicado, rank == índice lógico da GPU.
    runner = InferenceRunner(config)
    runner.setup(rank, world_size, rank)
    runner.run()

def run_inference(
    input_path: str,
    output_path: str,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    gpus: int | list[int] | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    recursive: bool = True,
    mnt_degrees: bool = True,
    threshold: float = 0.5,
    compile_model: bool = False,
    max_image_dim: int = 1024,
    strategy: str = 'hybrid',
    num_cpu_workers: int = 4,
    quality_mask: bool = False,
    unmodulated: bool = False,
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

    Example:
        >>> run_inference('images/', 'output/', gpus=2, batch_size=8)
    """
    if full:
        quality_mask = True
        unmodulated = True

    image_paths = find_image_paths(input_path, recursive)

    # Determine the base path for preserving directory structure
    # If input_path is a directory, use it as base
    # If it's a file or list, use its parent directory
    if os.path.isdir(input_path):
        input_base_path = input_path
    elif os.path.isfile(input_path):
        input_base_path = os.path.dirname(input_path)
    else:
        input_base_path = None

    # Coleta toda a configuração em um único dicionário para facilitar
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
    """
    Função alvo para um worker thread. Recebe um bloco de resultados e os salva em disco.
    """
    # Usamos uma descrição simples se não for um processo DDP
    desc = f"Saving (Worker {worker_rank})" if worker_rank >= 0 else "Saving Results"
    
    # O worker não deve mostrar uma barra de progresso individual, pois várias podem ser executadas
    # em paralelo. Apenas iteramos e salvamos.
    for result_item in results_chunk:
        try:
            save_results(result_item, output_path, mnt_degrees, input_base_path)
        except Exception as e:
            # Log de erro é importante em threads para não falhar silenciosamente
            base_name = os.path.basename(result_item.get('input_path', 'unknown_file'))
            logger.warning(f"Failed to save result for {base_name} in chunk. Error: {e}")


class InferenceRunner:
    def __init__(self, config: dict):
        """
        Inicializa o runner com toda a configuração necessária.
        'config' é um dicionário contendo tudo: image_paths, output_path, batch_size, etc.
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
        """
        Configura o ambiente para este processo específico (worker).
        Isso inclui DDP, device, modelo e dataloader.
        
        Args:
            rank: Rank do processo no DDP (-1 para single process)
            world_size: Número total de processos
            gpu_id: Índice lógico da GPU a usar (para DDP)
        """
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank <= 0) # Rank 0 para DDP, -1 para single/cpu

        # 1. Configurar DDP e Device
        if self.world_size > 1:
            setup_ddp(self.rank, self.world_size, gpu_id)
            # Usa o índice lógico correto após remapeamento do CUDA_VISIBLE_DEVICES
            self.device = f"cuda:{gpu_id}"
        elif self.config['gpus'] and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        if self.is_main_process:
            logger.info(f"Setting up runner on device: {self.device}")
            create_output_directories(
                self.config['output_path'],
                quality_mask=self.config.get('quality_mask', False),
                unmodulated=self.config.get('unmodulated', False)
            )
        
        if self.world_size > 1:
            dist.barrier() # Garante que as pastas foram criadas

        # 2. Carregar o Modelo
        self.model = get_fingernet(
            weights_path=self.config['weights_path'],
            device=self.device
        )
        if self.config['compile_model']:
            if self.is_main_process: logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        # O DDP wrapper não é necessário para inferência, simplifica o código.

        # 3. Preparar Dataset e DataLoader
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

        # 4. Definir a estratégia de execução
        self.strategy = self.config['strategy']

    def run(self):
        """Despacha para o método de execução correto baseado na estratégia."""
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
        """Executa o pipeline otimizado para throughput (GPU-infer, CPU-postproc)."""
        # Esta é a lógica que já tínhamos, usando ThreadPoolExecutor.
        if self.is_main_process:
            logger.info("Starting inference loop...")

        num_cpu_workers = self.config['num_cpu_workers']
        
        with ThreadPoolExecutor(max_workers=num_cpu_workers) as executor:
            futures = []
            max_queue_size = 2 * num_cpu_workers

            with torch.no_grad():
                desc = f"GPU {self.rank}" if self.world_size > 1 else "Processing"
                iterator = tqdm(self.dataloader, desc=desc, disable=not self.is_main_process)
                
                for batch_tensors, batch_paths, batch_orig_shapes in iterator:
                    if batch_tensors is None: continue

                    # --- ETAPA GPU/CPU: INFERÊNCIA ---
                    _, _, padded_h, padded_w = batch_tensors.shape
                    batch_tensors = batch_tensors.to(self.device)
                    raw_outputs = self.model.fingernet(batch_tensors) # Chama o modelo core

                    # --- ETAPA DE TRANSFERÊNCIA (se necessário) ---
                    raw_outputs_cpu = {k: v.detach().cpu() for k, v in raw_outputs.items()}

                    # --- ETAPA DE SUBMISSÃO PARA CPU WORKERS ---
                    future = executor.submit(
                        postprocess_and_save_batch,
                        raw_outputs_cpu, batch_paths, batch_orig_shapes,
                        (padded_h, padded_w), self.config['output_path'], self.config['mnt_degrees'],
                        self.config.get('input_base_path'),
                        self.config.get('quality_mask', False),
                        self.config.get('unmodulated', False),
                        self.config.get('threshold', 0.5),
                    )
                    futures.append(future)

                    # Gerenciamento da fila para evitar sobrecarga de memória
                    if len(futures) >= max_queue_size:
                        futures.pop(0).result() # Espera o mais antigo terminar

            # Aguardar finalização de todas as tarefas
            if self.is_main_process:
                logger.info("Inference complete. Finalizing post-processing...")
            for future in tqdm(futures, desc=f"Finalizing (Worker {self.rank})", disable=not self.is_main_process):
                future.result()

        
    def _run_full_gpu(self):
        """
        Executa o pipeline otimizado para latência (tudo na GPU) com salvamento assíncrono.
        """
        # Use o mesmo número de workers da CPU que a estratégia híbrida para consistência
        num_save_workers = self.config['num_cpu_workers']
        
        # O tamanho do bloco que aciona o salvamento. Ex: 4 (batch_size) * 10 = 40 imagens.
        # Isso evita despachar tarefas muito pequenas para os workers.
        chunk_size = self.config['batch_size'] * 10 

        # Pool de workers para salvar os resultados em segundo plano
        with ThreadPoolExecutor(max_workers=num_save_workers) as save_executor:
            futures = []
            chunk_para_salvar = []

            with torch.no_grad():
                desc = f"GPU {self.rank}" if self.world_size > 1 else "Processing"
                iterator = tqdm(self.dataloader, desc=desc, disable=not self.is_main_process)
                
                for batch_tensors, batch_paths, batch_orig_shapes in iterator:
                    if batch_tensors is None: continue

                    # --- ETAPA GPU: Inferência E Pós-processamento ---
                    batch_tensors = batch_tensors.to(self.device)
                    _qm = self.config.get('quality_mask', False)
                    _um = self.config.get('unmodulated', False)
                    final_outputs = self.model(batch_tensors, minutiae_threshold=self.config.get('threshold', 0.5), quality_mask=_qm, unmodulated=_um)

                    # --- ETAPA DE TRANSFERÊNCIA E COLETA (RÁPIDA) ---
                    for i in range(len(batch_paths)):
                        orig_h, orig_w = batch_orig_shapes[0][i].item(), batch_orig_shapes[1][i].item()
                        result_item = {
                            'input_path': batch_paths[i],
                            'minutiae': final_outputs['minutiae'][i].cpu().numpy(),
                            'enhanced_image': final_outputs['enhanced_image'][i][:orig_h, :orig_w].cpu().numpy(),
                            'segmentation_mask': final_outputs['segmentation_mask'][i][:orig_h, :orig_w].cpu().numpy(),
                            'orientation_field': final_outputs['orientation_field'][i][:orig_h, :orig_w].cpu().numpy(),
                        }
                        if _qm and 'quality_mask' in final_outputs:
                            result_item['quality_mask'] = final_outputs['quality_mask'][i][:orig_h, :orig_w].cpu().numpy()
                        if _um:
                            result_item['enhanced_image_unmod'] = final_outputs['enhanced_image_unmod'][i][:orig_h, :orig_w].cpu().numpy()
                            result_item['orientation_field_unmod'] = final_outputs['orientation_field_unmod'][i][:orig_h, :orig_w].cpu().numpy()
                            result_item['minutiae_unmod'] = final_outputs['minutiae_unmod'][i].cpu().numpy()
                        chunk_para_salvar.append(result_item)

                    # --- ETAPA DE DESPACHO PARA WORKERS ---
                    if len(chunk_para_salvar) >= chunk_size:
                        future = save_executor.submit(
                            _save_results_chunk,
                            chunk_para_salvar,
                            self.config['output_path'],
                            self.config['mnt_degrees'],
                            self.rank,
                            self.config.get('input_base_path')
                        )
                        futures.append(future)
                        chunk_para_salvar = []  # Limpa o bloco para o próximo ciclo

            # --- FINALIZAÇÃO ---
            # Despacha o último bloco, que pode ser menor que o chunk_size
            if chunk_para_salvar:
                future = save_executor.submit(
                    _save_results_chunk,
                    chunk_para_salvar,
                    self.config['output_path'],
                    self.config['mnt_degrees'],
                    self.rank,
                    self.config.get('input_base_path')
                )
                futures.append(future)

            # Aguarda todos os workers de salvamento terminarem seu trabalho
            if self.is_main_process:
                logger.info("Inference complete. Waiting for save workers to finish...")
            
            # Usamos uma barra de progresso para esperar os futuros, dando feedback ao usuário
            for future in tqdm(futures, desc=f"Finalizing Save (Worker {self.rank})", disable=not self.is_main_process):
                future.result() # .result() espera a conclusão e levanta exceções se houver

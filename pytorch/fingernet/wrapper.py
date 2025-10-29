import torch
from torch import nn
import torch.nn.functional as F
from .model import FingerNet
from .fnet_utils import get_fingernet_logger, FnetTimer, logging, DEFAULT_WEIGHTS_PATH, DEFAULT_DEVICE
import kornia
import os
import numpy as np

logger = get_fingernet_logger(__name__, level=logging.INFO)

class FingerNetWrapper(nn.Module):
    def __init__(self, model: FingerNet):
        super().__init__()
        self.fingernet = model

    def forward(self, x: torch.Tensor, minutiae_threshold: float = 0.5) -> dict[str, torch.Tensor]:
        
        padded_x = self.preprocess(x)
        
        with torch.no_grad():
            raw_outputs = self.fingernet(padded_x) 

        post_x = self.postprocess(raw_outputs, minutiae_threshold)

        return post_x

    def time(self, x: torch.Tensor, minutiae_threshold: float = 0.5) -> dict[str, torch.Tensor]:
        
        padded_x = self.preprocess(x)
        
        with torch.no_grad():
            with FnetTimer("Full Inference", logger):
                raw_outputs = self.fingernet.time(padded_x)

        with FnetTimer("Post-processing", logger):
            post_x = self.postprocess_time(raw_outputs, minutiae_threshold)

        return post_x

    def prepare_input(self, x: np.ndarray) -> torch.Tensor:
        """Converts a numpy image to a torch tensor suitable for the model."""
        # Check if input is 2D (H, W)
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)  # add channel dimension
            x = np.expand_dims(x, axis=0)  # add batch dimension
        if x.ndim == 3:
            # This could be (C, H, W) or (B, H, W).
            # We assume (B, H, W) if B > 1
            if x.shape[0] > 1:
                x = np.expand_dims(x, axis=1)  # add channel dimension
            else:
                x = np.expand_dims(x, axis=0)  # add batch dimension
        if x.ndim == 4:
            tensor_x = torch.tensor(x, dtype=torch.float32)
        else:
            raise ValueError("Input numpy array must be 2D, 3D - with Channel, or 4D - with Batch.")
        
        # Detect device
        device = next(self.fingernet.parameters()).device
        tensor_x = tensor_x.to(device)
        return tensor_x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        return F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

    def postprocess(self, outputs: dict, threshold: float) -> dict[str, torch.Tensor]:
        return postprocess(outputs, threshold)

    def postprocess_time(self, outputs: dict, threshold: float) -> dict[str, torch.Tensor]:
        return postprocess_time(outputs, threshold)

def get_fingernet(weights_path: str = DEFAULT_WEIGHTS_PATH, device: str = DEFAULT_DEVICE) -> FingerNetWrapper:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    logger.info(f"Selected device: {device}")
    logger.info("Loading FingerNet architecture...")
    fingernet_model = FingerNet()
    logger.info(f"Loading weights from: {weights_path}")
    fingernet_model.load_state_dict(torch.load(weights_path, map_location=device))
    fingernet_model.eval()

    logger.info("Creating and moving the wrapper to the device...")
    fnet_wrapper = FingerNetWrapper(model=fingernet_model).to(device)

    if device == "cpu":
        logger.info("Moving wrapper to channels_last memory format...")
        fnet_wrapper.to(memory_format=torch.channels_last)

    logger.info("Model ready for inference.")
    
    return fnet_wrapper

def postprocess(outputs: dict, threshold: float) -> dict[str, torch.Tensor]:
    # 1. Binarização e limpeza da máscara de segmentação
    cleaned_mask = _post_binarize_mask_fast(outputs['segmentation'])
    cleaned_mask_up = torch.nn.functional.interpolate(
        cleaned_mask.unsqueeze(1).float(),
        scale_factor=8,
        mode='nearest'
    ).squeeze(1)

    # 2. Detecção de minúcias (incluindo NMS)
    # O resultado é uma lista de tensores, um para cada imagem no lote.
    final_minutiae_list = _post_detect_minutiae(outputs, cleaned_mask, threshold)

    # 3. Processamento do campo de orientação
    ori = outputs['orientation']
    ori_idx = torch.argmax(ori, dim=1)
    ori_idx_up = torch.nn.functional.interpolate(
        ori_idx.unsqueeze(1).float(),
        scale_factor=8,
        mode='nearest'
    ).squeeze(1)
    orientation_field = (ori_idx_up * 2.0 - 89.) * torch.pi / 180.0
    orientation_field = orientation_field * cleaned_mask_up

    # 4. Processamento da imagem melhorada
    enh_real = outputs['enhanced_real'].squeeze(1)
    enh_real = enh_real * cleaned_mask_up
    
    # Normalização Min-Max para visualização
    b, h, w = enh_real.shape
    enh_flat = enh_real.view(b, -1)
    enh_min = enh_flat.min(dim=1, keepdim=True)[0]
    enh_max = enh_flat.max(dim=1, keepdim=True)[0]
    enh_norm = (enh_flat - enh_min) / (enh_max - enh_min + 1e-8)
    enh_visual = (enh_norm.view(b, h, w) * 255).byte()

    return {
        'minutiae': final_minutiae_list,
        'enhanced_image': enh_visual,
        'segmentation_mask': (cleaned_mask_up * 255).byte(),
        'orientation_field': orientation_field
    }

def postprocess_time(outputs: dict, threshold: float) -> dict[str, torch.Tensor]:
    # 1. Binarização e limpeza da máscara de segmentação
    with FnetTimer("Mask Binarization and Cleaning", logger):
        cleaned_mask = _post_binarize_mask_fast(outputs['segmentation'])
        cleaned_mask_up = torch.nn.functional.interpolate(
            cleaned_mask.unsqueeze(1).float(),
            scale_factor=8,
            mode='nearest'
        ).squeeze(1)

    # 2. Detecção de minúcias (incluindo NMS)
    # O resultado é uma lista de tensores, um para cada imagem no lote.
    with FnetTimer("Minutiae Detection", logger):
        final_minutiae_list = _post_detect_minutiae(outputs, cleaned_mask, threshold)

    # 3. Processamento do campo de orientação
    with FnetTimer("Orientation Field Processing", logger):
        ori = outputs['orientation']
        ori_idx = torch.argmax(ori, dim=1)
        ori_idx_up = torch.nn.functional.interpolate(
            ori_idx.unsqueeze(1).float(),
            scale_factor=8,
            mode='nearest'
        ).squeeze(1)
        orientation_field = (ori_idx_up * 2.0 - 89.) * torch.pi / 180.0
        orientation_field = orientation_field * cleaned_mask_up

    # 4. Processamento da imagem melhorada
    with FnetTimer("Enhanced Image Processing", logger):
        enh_real = outputs['enhanced_real'].squeeze(1)
        enh_real = enh_real * cleaned_mask_up
    
    # Normalização Min-Max para visualização
    with FnetTimer("Enhanced Image Normalization", logger):
        b, h, w = enh_real.shape
        enh_flat = enh_real.view(b, -1)
        enh_min = enh_flat.min(dim=1, keepdim=True)[0]
        enh_max = enh_flat.max(dim=1, keepdim=True)[0]
        enh_norm = (enh_flat - enh_min) / (enh_max - enh_min + 1e-8)
        enh_visual = (enh_norm.view(b, h, w) * 255).byte()

    return {
        'minutiae': final_minutiae_list,
        'enhanced_image': enh_visual,
        'segmentation_mask': (cleaned_mask_up * 255).byte(),
        'orientation_field': orientation_field
    }


def _post_binarize_mask(self, seg_map: torch.Tensor) -> torch.Tensor:
    """Binariza e limpa a máscara de segmentação usando Kornia."""
    seg_map_squeezed = seg_map.squeeze(1)
    binarized = torch.round(seg_map_squeezed).bool()
    # # Kornia espera um shape [B, C, H, W], por isso o unsqueeze/squeeze
    kernel = torch.ones(5, 5, device=seg_map.device)
    cleaned = kornia.morphology.opening(binarized.unsqueeze(1), kernel).squeeze(1)
    return cleaned

def gaussian_blur_torch(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    def _get_gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords -= kernel_size // 2
        # avoid using tensor ** operations (pow) which can trigger symbolic
        # interpretation inside torch.compile / torch._dynamo (sympy interp)
        coords_sq = coords * coords
        sigma_sq = sigma * sigma
        g = torch.exp(-(coords_sq) / (2 * sigma_sq))
        g /= g.sum()
        return g
    
    # 1. Obter o kernel 1D
    kernel_1d = _get_gaussian_kernel1d(kernel_size, sigma, device=image.device, dtype=image.dtype)
    
    # 2. Obter o número de canais para aplicar o blur em cada um independentemente
    B, C, H, W = image.shape
    
    # 3. Preparar kernels para convolução horizontal e vertical
    # A conv2d espera um shape [out_channels, in_channels/groups, kH, kW]
    # Usamos `groups=C` para que cada canal seja convolvido com seu próprio kernel.
    kernel_h = kernel_1d.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)

    # 4. Calcular o padding para manter o tamanho da imagem
    padding = kernel_size // 2
    
    # 5. Aplicar a convolução horizontal
    blurred_h = F.conv2d(image, kernel_h, padding=(0, padding), groups=C)
    
    # 6. Aplicar a convolução vertical no resultado da horizontal
    blurred_hv = F.conv2d(blurred_h, kernel_v, padding=(padding, 0), groups=C)
    
    return blurred_hv

def _post_binarize_mask_fast(seg_map: torch.Tensor) -> torch.Tensor:
    """
    Binariza e limpa a máscara de segmentação de forma extremamente rápida,
    usando uma implementação de blur Gaussiano apenas com PyTorch.
    """
    # 1. Binariza a máscara de entrada para valores 0.0 ou 1.0
    binarized_float = torch.round(seg_map.squeeze(1))
    
    # 2. Adiciona a dimensão de canal para a convolução
    # Shape: [B, H, W] -> [B, 1, H, W]
    image_with_channel = binarized_float.unsqueeze(1)
    
    # 3. Aplica o blur Gaussiano rápido
    blurred = gaussian_blur_torch(image_with_channel, kernel_size=5, sigma=1.5)
    
    # 4. Re-binariza o resultado para obter a máscara final e limpa.
    cleaned_mask = torch.round(blurred)

    return cleaned_mask.squeeze(1)

def _post_detect_minutiae(outputs: dict, cleaned_mask: torch.Tensor, threshold: float) -> list:
    """Detecta, filtra e aplica NMS nas minúcias para um lote inteiro."""
    mnt_score_batch = outputs['minutiae_score'].squeeze(1) * cleaned_mask
    mnt_orient_batch = outputs['minutiae_orientation']
    mnt_x_offset_batch = outputs['minutiae_x_offset']
    mnt_y_offset_batch = outputs['minutiae_y_offset']
    
    batch_size = mnt_score_batch.shape[0]
    final_minutiae_list = []

    for i in range(batch_size):
        # Encontra coordenadas das minúcias acima do limiar
        rows, cols = torch.where(mnt_score_batch[i] > threshold)
        if rows.shape[0] == 0:
            final_minutiae_list.append(torch.empty((0, 4), device=mnt_score_batch.device))
            continue

        # Extrai scores, ângulos e offsets
        scores = mnt_score_batch[i][rows, cols]
        angles_idx = torch.argmax(mnt_orient_batch[i, :, rows, cols], dim=0)
        x_offsets = torch.argmax(mnt_x_offset_batch[i, :, rows, cols], dim=0)
        y_offsets = torch.argmax(mnt_y_offset_batch[i, :, rows, cols], dim=0)
        
        # Calcula valores finais
        angles = (angles_idx * 2.0 - 89.0) * (torch.pi / 180.0)
        x_coords = cols * 8.0 + x_offsets
        y_coords = rows * 8.0 + y_offsets
        
        minutiae_raw = torch.stack([x_coords, y_coords, angles, scores], dim=-1)
        
        # Aplica NMS
        final_minutiae = _post_nms(minutiae_raw)
        final_minutiae_list.append(final_minutiae)
        
    return final_minutiae_list

def _post_nms(minutiae: torch.Tensor, dist_thresh: float = 16.0, angle_thresh: float = torch.pi/6) -> torch.Tensor:
    """Aplica Non-Maximum Suppression (NMS) em um tensor de minúcias."""
    if minutiae.shape[0] == 0:
        return minutiae

    # Ordena por score (decrescente)
    order = torch.argsort(minutiae[:, 3], descending=True)
    minutiae = minutiae[order]

    # Calcula matriz de distância Euclidiana e angular
    dist_matrix = torch.cdist(minutiae[:, :2], minutiae[:, :2])
    
    # Cálculo da distância angular via broadcasting
    angles1 = minutiae[:, 2].unsqueeze(1) # [N, 1]
    angles2 = minutiae[:, 2].unsqueeze(0) # [1, N]
    angle_delta = torch.abs(angles1 - angles2)
    angle_matrix = torch.minimum(angle_delta, 2 * torch.pi - angle_delta)

    # Máscara para supressão: True onde a distância E o ângulo são menores que o limiar
    suppress_mask = (dist_matrix < dist_thresh) & (angle_matrix < angle_thresh)
    
    keep = torch.ones(minutiae.shape[0], dtype=torch.bool, device=minutiae.device)
    for i in range(minutiae.shape[0]):
        if keep[i]:
            # Suprime todos os outros pontos que estão muito próximos deste
            # torch.where retorna uma tupla, pegamos o primeiro elemento
            suppress_indices = torch.where(suppress_mask[i, i+1:])[0]
            keep[i + 1 + suppress_indices] = False
            
    return minutiae[keep]
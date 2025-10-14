import torch
import time
import kornia
import numpy as np

def postprocess(outputs: dict, threshold: float) -> dict[str, torch.Tensor]:
    # 1. Binarização e limpeza da máscara de segmentação
    t1 = time.time()
    cleaned_mask = _post_binarize_mask(outputs['segmentation'])
    cleaned_mask_up = _post_binarize_mask(outputs['segmentation upsample'], upsample_factor=8)
    t2 = time.time()

    # 2. Detecção de minúcias (incluindo NMS)
    # O resultado é uma lista de tensores, um para cada imagem no lote.
    final_minutiae_list = _post_detect_minutiae(outputs, cleaned_mask, threshold)
    #final_minutiae_list = parallel_matrix_nms(outputs, cleaned_mask, threshold)
    t3 = time.time()
    
    # 3. Processamento do campo de orientação
    ori_up = outputs['orientation upsample']
    orientation_field = (torch.argmax(ori_up, dim=1).float() * 2.0 - 90.) * torch.pi / 180.0
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
    t4 = time.time()

    #print(f"Postprocess - Mask: {t2 - t1:.4f}s, Minutiae: {t3 - t2:.4f}s, Orientation & Enh: {t4 - t3:.4f}s, Total= {t4 - t1:.4f}s")

    return {
        'minutiae': final_minutiae_list,
        'enhanced_image': enh_visual,
        'segmentation_mask': (cleaned_mask_up * 255).byte(),
        'orientation_field': orientation_field
    }

def _post_binarize_mask(seg_map: torch.Tensor, upsample_factor: int = 1) -> torch.Tensor:
    """Binariza e limpa a máscara de segmentação usando Kornia."""
    seg_map_squeezed = seg_map.squeeze(1)
    binarized = torch.round(seg_map_squeezed)
    kernel = torch.ones(5 * upsample_factor, 5 * upsample_factor, device=seg_map.device)
    # Kornia espera um shape [B, C, H, W], por isso o unsqueeze/squeeze
    cleaned = kornia.morphology.opening(binarized.unsqueeze(1), kernel).squeeze(1)
    return cleaned

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

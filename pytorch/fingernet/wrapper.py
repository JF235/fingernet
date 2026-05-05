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

    def forward(self, x: torch.Tensor, minutiae_threshold: float = 0.05,
                quality_mask: bool = False, full: bool = False) -> dict[str, torch.Tensor]:

        padded_x = self.preprocess(x)

        with torch.inference_mode():
            raw_outputs = self.fingernet(padded_x)

        post_x = self.postprocess(raw_outputs, minutiae_threshold,
                                  quality_mask=quality_mask, full=full)

        return post_x

    def time(self, x: torch.Tensor, minutiae_threshold: float = 0.05,
             quality_mask: bool = False, full: bool = False) -> dict[str, torch.Tensor]:

        padded_x = self.preprocess(x)

        with torch.no_grad():
            with FnetTimer("Full Inference", logger):
                raw_outputs = self.fingernet.time(padded_x)

        with FnetTimer("Post-processing", logger):
            post_x = self.postprocess_time(raw_outputs, minutiae_threshold,
                                           quality_mask=quality_mask, full=full)

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

    def postprocess(self, outputs: dict, threshold: float,
                    quality_mask: bool = False, full: bool = False) -> dict[str, torch.Tensor]:
        return postprocess(outputs, threshold, quality_mask=quality_mask, full=full)

    def postprocess_time(self, outputs: dict, threshold: float,
                         quality_mask: bool = False, full: bool = False) -> dict[str, torch.Tensor]:
        return postprocess_time(outputs, threshold, quality_mask=quality_mask, full=full)

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

def _normalize_minmax_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize per image in the batch, return uint8 in [0, 255]."""
    b, h, w = x.shape
    flat = x.view(b, -1)
    mn = flat.min(dim=1, keepdim=True)[0]
    mx = flat.max(dim=1, keepdim=True)[0]
    norm = (flat - mn) / (mx - mn + 1e-8)
    return (norm.view(b, h, w) * 255).byte()


def postprocess(outputs: dict, threshold: float,
                quality_mask: bool = False, full: bool = False) -> dict[str, torch.Tensor]:
    """Post-process FingerNet raw outputs.

    Default outputs (always present):
        - `minutiae`           — list[B] of [N, 4] (x, y, angle rad, score),
                                 already filtered by the mask and with NMS.
        - `segmentation_mask`  — binary mask uint8 [0, 255].
        - `orientation_field`  — orientation field WITHOUT mask modulation (rad).
        - `enhanced_image`     — enhanced image WITHOUT mask modulation, uint8 [0, 255].

    Opt-in extras:
        - `quality_mask=True`  → adds `quality_mask` (continuous sigmoid).
        - `full=True`          → adds all the above + modulated versions
                                 (`*_mod`) and `minutiae_unmod` (without
                                 mask filter). Implies `quality_mask=True`.
    """
    # 1. Binarized + smoothed mask
    cleaned_mask = _post_binarize_mask_fast(outputs['segmentation'])
    cleaned_mask_up = torch.nn.functional.interpolate(
        cleaned_mask.unsqueeze(1).float(),
        scale_factor=8,
        mode='nearest'
    ).squeeze(1)

    # 2. Minutiae (always the mask-filtered ones; minutiae_unmod only if full=True)
    if full:
        minutiae_list, minutiae_unmod_list = _post_detect_minutiae(
            outputs, cleaned_mask, threshold, unmodulated=True)
    else:
        minutiae_list = _post_detect_minutiae(outputs, cleaned_mask, threshold)

    # 3. Orientation field (raw / unmodulated by default)
    ori_idx = torch.argmax(outputs['orientation'], dim=1)
    ori_idx_up = torch.nn.functional.interpolate(
        ori_idx.unsqueeze(1).float(), scale_factor=8, mode='nearest'
    ).squeeze(1)
    orientation_field_raw = (ori_idx_up * 2.0 - 89.) * torch.pi / 180.0

    # 4. Enhanced image (raw / unmodulated by default)
    enh_real_raw = outputs['enhanced_real'].squeeze(1)
    enhanced_image_raw = _normalize_minmax_to_uint8(enh_real_raw)

    result = {
        'minutiae': minutiae_list,
        'enhanced_image': enhanced_image_raw,
        'segmentation_mask': (cleaned_mask_up * 255).byte(),
        'orientation_field': orientation_field_raw,
    }

    if quality_mask or full:
        seg_continuous_up = torch.nn.functional.interpolate(
            outputs['segmentation'], scale_factor=8, mode='bilinear', align_corners=False
        ).squeeze(1)
        result['quality_mask'] = (seg_continuous_up * 255).byte()

    if full:
        result['orientation_field_mod'] = orientation_field_raw * cleaned_mask_up
        result['enhanced_image_mod'] = _normalize_minmax_to_uint8(
            enh_real_raw * cleaned_mask_up
        )
        result['minutiae_unmod'] = minutiae_unmod_list

    return result


def postprocess_time(outputs: dict, threshold: float,
                     quality_mask: bool = False, full: bool = False) -> dict[str, torch.Tensor]:
    """Instrumented version of `postprocess` (DEBUG logger)."""
    with FnetTimer("Mask Binarization and Cleaning", logger):
        cleaned_mask = _post_binarize_mask_fast(outputs['segmentation'])
        cleaned_mask_up = torch.nn.functional.interpolate(
            cleaned_mask.unsqueeze(1).float(), scale_factor=8, mode='nearest'
        ).squeeze(1)

    with FnetTimer("Minutiae Detection", logger):
        if full:
            minutiae_list, minutiae_unmod_list = _post_detect_minutiae(
                outputs, cleaned_mask, threshold, unmodulated=True)
        else:
            minutiae_list = _post_detect_minutiae(outputs, cleaned_mask, threshold)

    with FnetTimer("Orientation Field Processing", logger):
        ori_idx = torch.argmax(outputs['orientation'], dim=1)
        ori_idx_up = torch.nn.functional.interpolate(
            ori_idx.unsqueeze(1).float(), scale_factor=8, mode='nearest'
        ).squeeze(1)
        orientation_field_raw = (ori_idx_up * 2.0 - 89.) * torch.pi / 180.0

    with FnetTimer("Enhanced Image Normalization", logger):
        enh_real_raw = outputs['enhanced_real'].squeeze(1)
        enhanced_image_raw = _normalize_minmax_to_uint8(enh_real_raw)

    result = {
        'minutiae': minutiae_list,
        'enhanced_image': enhanced_image_raw,
        'segmentation_mask': (cleaned_mask_up * 255).byte(),
        'orientation_field': orientation_field_raw,
    }

    if quality_mask or full:
        seg_continuous_up = torch.nn.functional.interpolate(
            outputs['segmentation'], scale_factor=8, mode='bilinear', align_corners=False
        ).squeeze(1)
        result['quality_mask'] = (seg_continuous_up * 255).byte()

    if full:
        result['orientation_field_mod'] = orientation_field_raw * cleaned_mask_up
        result['enhanced_image_mod'] = _normalize_minmax_to_uint8(
            enh_real_raw * cleaned_mask_up
        )
        result['minutiae_unmod'] = minutiae_unmod_list

    return result


def _post_binarize_mask(self, seg_map: torch.Tensor) -> torch.Tensor:
    """Binarize and clean the segmentation mask using Kornia."""
    seg_map_squeezed = seg_map.squeeze(1)
    binarized = torch.round(seg_map_squeezed).bool()
    # Kornia expects shape [B, C, H, W], hence the unsqueeze/squeeze.
    kernel = torch.ones(5, 5, device=seg_map.device)
    cleaned = kornia.morphology.opening(binarized.unsqueeze(1), kernel).squeeze(1)
    return cleaned

def gaussian_blur_torch(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    def _get_gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords -= kernel_size // 2
        # Avoid tensor ** (pow), which can trigger symbolic interpretation
        # inside torch.compile / torch._dynamo (sympy interp).
        coords_sq = coords * coords
        sigma_sq = sigma * sigma
        g = torch.exp(-(coords_sq) / (2 * sigma_sq))
        g /= g.sum()
        return g

    # 1. Build the 1-D kernel
    kernel_1d = _get_gaussian_kernel1d(kernel_size, sigma, device=image.device, dtype=image.dtype)

    # 2. Number of channels (each channel is blurred independently)
    B, C, H, W = image.shape

    # 3. Build horizontal/vertical conv kernels.
    # conv2d expects [out_channels, in_channels/groups, kH, kW]. We use
    # groups=C so that every channel is convolved with its own kernel.
    kernel_h = kernel_1d.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)

    # 4. Padding to keep the spatial size constant
    padding = kernel_size // 2

    # 5. Horizontal pass
    blurred_h = F.conv2d(image, kernel_h, padding=(0, padding), groups=C)

    # 6. Vertical pass on the horizontal result
    blurred_hv = F.conv2d(blurred_h, kernel_v, padding=(padding, 0), groups=C)

    return blurred_hv

def _post_binarize_mask_fast(seg_map: torch.Tensor) -> torch.Tensor:
    """Fast binarization + cleanup of the segmentation mask, using only PyTorch ops."""
    # 1. Binarize input to {0.0, 1.0}
    binarized_float = torch.round(seg_map.squeeze(1))

    # 2. Add channel dim for conv: [B, H, W] -> [B, 1, H, W]
    image_with_channel = binarized_float.unsqueeze(1)

    # 3. Fast separable Gaussian blur
    blurred = gaussian_blur_torch(image_with_channel, kernel_size=5, sigma=1.5)

    # 4. Re-binarize for the final cleaned mask
    cleaned_mask = torch.round(blurred)

    return cleaned_mask.squeeze(1)

def _post_detect_minutiae_single(mnt_score, mnt_orient_batch_i, mnt_x_offset_batch_i,
                                  mnt_y_offset_batch_i, threshold, device):
    """Detect and NMS minutiae for a single image."""
    rows, cols = torch.where(mnt_score > threshold)
    if rows.shape[0] == 0:
        return torch.empty((0, 4), device=device)

    scores = mnt_score[rows, cols]
    angles_idx = torch.argmax(mnt_orient_batch_i[:, rows, cols], dim=0)
    x_offsets = torch.argmax(mnt_x_offset_batch_i[:, rows, cols], dim=0)
    y_offsets = torch.argmax(mnt_y_offset_batch_i[:, rows, cols], dim=0)

    angles = (angles_idx * 2.0 - 89.0) * (torch.pi / 180.0)
    x_coords = cols * 8.0 + x_offsets
    y_coords = rows * 8.0 + y_offsets

    minutiae_raw = torch.stack([x_coords, y_coords, angles, scores], dim=-1)
    return _post_nms(minutiae_raw)


def _post_detect_minutiae(outputs: dict, cleaned_mask: torch.Tensor, threshold: float,
                          unmodulated: bool = False):
    """Detect, filter, and NMS minutiae for a full batch."""
    mnt_score_batch = outputs['minutiae_score'].squeeze(1) * cleaned_mask
    mnt_orient_batch = outputs['minutiae_orientation']
    mnt_x_offset_batch = outputs['minutiae_x_offset']
    mnt_y_offset_batch = outputs['minutiae_y_offset']

    if unmodulated:
        mnt_score_batch_unmod = outputs['minutiae_score'].squeeze(1)

    batch_size = mnt_score_batch.shape[0]
    final_minutiae_list = []
    final_minutiae_unmod_list = [] if unmodulated else None

    for i in range(batch_size):
        final_minutiae_list.append(
            _post_detect_minutiae_single(
                mnt_score_batch[i], mnt_orient_batch[i],
                mnt_x_offset_batch[i], mnt_y_offset_batch[i],
                threshold, mnt_score_batch.device))

        if unmodulated:
            final_minutiae_unmod_list.append(
                _post_detect_minutiae_single(
                    mnt_score_batch_unmod[i], mnt_orient_batch[i],
                    mnt_x_offset_batch[i], mnt_y_offset_batch[i],
                    threshold, mnt_score_batch.device))

    if unmodulated:
        return final_minutiae_list, final_minutiae_unmod_list
    return final_minutiae_list

def _post_nms(minutiae: torch.Tensor, dist_thresh: float = 16.0, angle_thresh: float = torch.pi/6) -> torch.Tensor:
    """Greedy Non-Maximum Suppression (NMS) on a minutiae tensor."""
    if minutiae.shape[0] == 0:
        return minutiae

    # Sort by score (descending)
    order = torch.argsort(minutiae[:, 3], descending=True)
    minutiae = minutiae[order]

    # Pairwise Euclidean and angular distance matrices.
    # Using the direct (a-b)^2 form instead of torch.cdist; cdist takes the
    # shortcut ||a-b||^2 = ||a||^2 + ||b||^2 - 2·a·b, which under TF32
    # suffers catastrophic cancellation for pixel-scale coordinates
    # (~10^3): every term has absolute error ~ ||a||^2 * 1e-3, comparable
    # to the actual result ~16^2 = 256, flipping NMS decisions near
    # dist_thresh. The elementwise form is bit-exact under TF32 at the
    # same cost (~0.18ms vs 0.17ms for N=2000, vs 6.5ms for
    # compute_mode='donot_use_mm_for_euclid_dist').
    xy = minutiae[:, :2]
    diff = xy.unsqueeze(0) - xy.unsqueeze(1)        # [N, N, 2]
    dist_matrix = torch.sqrt((diff * diff).sum(-1))  # [N, N]

    # Angular distance via broadcasting
    angles1 = minutiae[:, 2].unsqueeze(1)  # [N, 1]
    angles2 = minutiae[:, 2].unsqueeze(0)  # [1, N]
    angle_delta = torch.abs(angles1 - angles2)
    angle_matrix = torch.minimum(angle_delta, 2 * torch.pi - angle_delta)

    # Suppression mask: True where BOTH distance and angle are below threshold
    suppress_mask = (dist_matrix < dist_thresh) & (angle_matrix < angle_thresh)

    # Greedy loop on CPU/NumPy. The original GPU loop cost ~75% of total
    # time because each `if keep[i]:` forces a GPU→host sync (minutiae per
    # image are in the hundreds–thousands ⇒ hundreds–thousands of syncs).
    # The suppression matrix is small (N×N bool, a few MB at most) and the
    # loop is bit-identical: only the execution location changes.
    suppress_cpu = suppress_mask.cpu().numpy()
    n = suppress_cpu.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if keep[i]:
            keep[i + 1:] &= ~suppress_cpu[i, i + 1:]
    keep_t = torch.from_numpy(keep).to(minutiae.device)
    return minutiae[keep_t]
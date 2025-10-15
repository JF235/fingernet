import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
import os
import kornia
from .fnet_utils import get_fingernet_logger, FnetTimer, logging, DEFAULT_DEVICE, DEFAULT_WEIGHTS_PATH

logger = get_fingernet_logger(__name__, level=logging.INFO)

class ImgNormalization(nn.Module):
    def __init__(self, m0=0.0, var0=1.0):
        super().__init__()
        self.m0 = m0
        self.var0 = var0
    def forward(self, x):
        m = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        var = torch.var(x, dim=(1, 2, 3), keepdim=True)
        after = torch.sqrt(self.var0 * torch.square(x - m) / (var + 1e-8))
        return torch.where(x > m, self.m0 + after, self.m0 - after)

class ConvBNPReLU(nn.Module):
    """Convolução -> BatchNorm -> PReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, 
            padding=padding, dilation=dilation, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99)
        self.prelu = nn.PReLU(num_parameters=out_channels, init=0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class FeatureExtractor(nn.Module):
    """Extrai características da imagem de entrada (Backbone VGG)."""
    def __init__(self):
        super().__init__()
        self.conv1_1 = ConvBNPReLU(1, 64, 3)
        self.conv1_2 = ConvBNPReLU(64, 64, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = ConvBNPReLU(64, 128, 3)
        self.conv2_2 = ConvBNPReLU(128, 128, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = ConvBNPReLU(128, 256, 3)
        self.conv3_2 = ConvBNPReLU(256, 256, 3)
        self.conv3_3 = ConvBNPReLU(256, 256, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1_1(x); x = self.conv1_2(x); x = self.pool1(x)
        x = self.conv2_1(x); x = self.conv2_2(x); x = self.pool2(x)
        x = self.conv3_1(x); x = self.conv3_2(x); x = self.conv3_3(x)
        return self.pool3(x)

class OrientationSegmentationHead(nn.Module):
    """Prediz orientação e segmentação a partir das características (ASPP)."""
    def __init__(self):
        super().__init__()
        self.atrous_1 = ConvBNPReLU(256, 256, 3, dilation=1)
        self.ori_branch_1 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 90, 1))
        self.seg_branch_1 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 1, 1))
        self.atrous_2 = ConvBNPReLU(256, 256, 3, dilation=4)
        self.ori_branch_2 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 90, 1))
        self.seg_branch_2 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 1, 1))
        self.atrous_3 = ConvBNPReLU(256, 256, 3, dilation=8)
        self.ori_branch_3 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 90, 1))
        self.seg_branch_3 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 1, 1))

    def forward(self, features):
        o1 = self.ori_branch_1(self.atrous_1(features)); s1 = self.seg_branch_1(self.atrous_1(features))
        o2 = self.ori_branch_2(self.atrous_2(features)); s2 = self.seg_branch_2(self.atrous_2(features))
        o3 = self.ori_branch_3(self.atrous_3(features)); s3 = self.seg_branch_3(self.atrous_3(features))
        ori_out = torch.sigmoid(o1 + o2 + o3)
        seg_out = torch.sigmoid(s1 + s2 + s3)
        return ori_out, seg_out

class EnhancementModule(nn.Module):
    """Realça a imagem usando filtros de Gabor e a orientação prevista."""
    def __init__(self):
        super().__init__()
        self.gabor_real = nn.Conv2d(1, 90, 25, padding='same', bias=True)
        self.gabor_imag = nn.Conv2d(1, 90, 25, padding='same', bias=True)

        # Pré-calcula o kernel gaussiano circular como tensor PyTorch
        length = 180
        stride = 2
        std = 3
        gaussian_pdf = signal.windows.gaussian(length + 1, std=std)
        y = np.reshape(np.arange(stride / 2, length, stride), [1, 1, -1, 1])
        label = np.reshape(np.arange(stride / 2, length, stride), [1, 1, 1, -1])
        delta = np.array(np.abs(label - y), dtype=int)
        delta = np.minimum(delta, length - delta) + length // 2
        glabel = gaussian_pdf[delta].astype(np.float32)
        # Salva como buffer para garantir que move junto com o módulo para o device
        self.register_buffer('glabel_tensor', torch.from_numpy(glabel).permute(2, 3, 0, 1))

    def _ori_highest_peak(self, y_pred, length=180, stride=2):
        """
        Aplica uma convolução 2D entre a predição y_pred e um kernel gaussiano circular,
        para detectar o pico de orientação dominante em dados angulares (ex: impressões digitais).
        O kernel é construído considerando a periodicidade dos ângulos (0-180 graus).
        """
        return F.conv2d(y_pred, self.glabel_tensor, padding='same')

    def _select_max_orientation(self, ori_map):
        """
        Given an orientation map tensor `ori_map` of shape (batch_size, num_orientations, height, width),
        this function normalizes the map by its maximum value along the orientation dimension, thresholds
        values close to the maximum (greater than 0.999), and returns a one-hot-like tensor indicating the
        positions of the maximum orientation for each spatial location.
        """
        max_vals, _ = torch.max(ori_map, dim=1, keepdim=True)
        x = ori_map / (max_vals + 1e-8); x = torch.where(x > 0.999, x, torch.zeros_like(x))
        return x / (torch.sum(x, dim=1, keepdim=True) + 1e-8)

    def _atan2(self, y, x):
        angle = torch.atan(y / (x + 1e-8))
        angle = torch.where(x > 0, angle, torch.zeros_like(x))
        angle = torch.where((x < 0) & (y >= 0), angle + np.pi, angle)
        angle = torch.where((x < 0) & (y < 0), angle - np.pi, angle)
        return angle

    def forward(self, original_image, ori_map):
        filtered_real = self.gabor_real(original_image)
        filtered_imag = self.gabor_imag(original_image)

        # Encontra o pico de orientação mais alto e seleciona a orientação máxima
        ori_map = self._ori_highest_peak(ori_map)
        ori_peak = self._select_max_orientation(ori_map)
        upsampled_ori = F.interpolate(ori_peak, scale_factor=8, mode='nearest')

        enh_real = torch.sum(filtered_real * upsampled_ori, dim=1, keepdim=True)
        enh_imag = torch.sum(filtered_imag * upsampled_ori, dim=1, keepdim=True)
        enhanced_phase = self._atan2(enh_imag, enh_real)

        return enh_real, enhanced_phase, upsampled_ori

class MinutiaeHead(nn.Module):
    """Bloco 4: Prediz os atributos das minúcias a partir da imagem realçada."""
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNPReLU(2, 64, 9); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBNPReLU(64, 128, 5); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBNPReLU(128, 256, 3); self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.o_branch = nn.Sequential(ConvBNPReLU(256 + 90, 256, 1), nn.Conv2d(256, 180, 1))
        self.w_branch = nn.Sequential(ConvBNPReLU(256, 256, 1), nn.Conv2d(256, 8, 1))
        self.h_branch = nn.Sequential(ConvBNPReLU(256, 256, 1), nn.Conv2d(256, 8, 1))
        self.s_branch = nn.Sequential(ConvBNPReLU(256, 256, 1), nn.Conv2d(256, 1, 1))

    def forward(self, enhanced_features, orientation_features):
        x = self.pool1(self.conv1(enhanced_features))
        x = self.pool2(self.conv2(x)); mnt_features = self.pool3(self.conv3(x))

        o_input = torch.cat([mnt_features, orientation_features], dim=1)
        mnt_o = torch.sigmoid(self.o_branch(o_input))
        mnt_w = torch.sigmoid(self.w_branch(mnt_features))
        mnt_h = torch.sigmoid(self.h_branch(mnt_features))
        mnt_s = torch.sigmoid(self.s_branch(mnt_features))

        return mnt_o, mnt_w, mnt_h, mnt_s

class FingerNet(nn.Module):
    """Modelo FingerNet completo, orquestrando a passagem de dados entre os blocos."""
    def __init__(self):
        super().__init__()
        self.img_norm = ImgNormalization()
        self.feature_extractor = FeatureExtractor()
        self.ori_seg_head = OrientationSegmentationHead()
        self.enhancement_module = EnhancementModule()
        self.minutiae_head = MinutiaeHead()

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna apenas o mapa de segmentação."""
        x_norm = self.img_norm(x)
        features = self.feature_extractor(x_norm)
        _, seg_map = self.ori_seg_head(features)
        upsampled_seg = F.interpolate(nn.functional.softsign(seg_map), scale_factor=8, mode='nearest')
        return upsampled_seg

    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna apenas a imagem realçada."""
        x_norm = self.img_norm(x)
        features = self.feature_extractor(x_norm)
        ori_map, _ = self.ori_seg_head(features)
        enh_real, _, _ = self.enhancement_module(x, ori_map)
        return enh_real

    def forward(self, x: torch.Tensor):
        """Define o fluxo de dados e retorna um dicionário com todas as saídas."""
        # Etapas do pipeline
        x_norm = self.img_norm(x)
        features = self.feature_extractor(x_norm)

        ori_map, seg_map = self.ori_seg_head(features)

        enh_real, enh_phase, upsampled_ori_map = self.enhancement_module(x, ori_map)

        upsampled_seg = F.interpolate(nn.functional.softsign(seg_map), scale_factor=8, mode='nearest')
        upsampled_seg_out = F.interpolate(seg_map, scale_factor=8, mode='nearest')

        minutiae_input = torch.cat([enh_phase, upsampled_seg], dim=1)
        
        mnt_o, mnt_w, mnt_h, mnt_s = self.minutiae_head(minutiae_input, ori_map)

        # Retorna um dicionário com saídas nomeadas para clareza
        return {
            'orientation upsample': upsampled_ori_map,
            'segmentation upsample': upsampled_seg_out,
            'segmentation': seg_map,
            'orientation': ori_map,
            'enhanced_real': enh_real,
            'enhanced_phase': enh_phase,
            'minutiae_orientation': mnt_o,
            'minutiae_x_offset': mnt_w,
            'minutiae_y_offset': mnt_h,
            'minutiae_score': mnt_s
        }

    def time(self, x: torch.Tensor):
        """Define o fluxo de dados e retorna um dicionário com todas as saídas."""
        # Etapas do pipeline
        
        x_norm = self.img_norm(x)

        with FnetTimer("Feature Extraction", logger):
            features = self.feature_extractor(x_norm)

        with FnetTimer("Orientation and Segmentation Head", logger):
            ori_map, seg_map = self.ori_seg_head(features)

        with FnetTimer("Enhancement Module", logger):
            enh_real, enh_phase, upsampled_ori_map = self.enhancement_module(x, ori_map)

        upsampled_seg = F.interpolate(nn.functional.softsign(seg_map), scale_factor=8, mode='nearest')
        upsampled_seg_out = F.interpolate(seg_map, scale_factor=8, mode='nearest')

        minutiae_input = torch.cat([enh_phase, upsampled_seg], dim=1)
        
        with FnetTimer("Minutiae Head", logger):
            mnt_o, mnt_w, mnt_h, mnt_s = self.minutiae_head(minutiae_input, ori_map)

        # Retorna um dicionário com saídas nomeadas para clareza
        return {
            'orientation upsample': upsampled_ori_map,
            'segmentation upsample': upsampled_seg_out,
            'segmentation': seg_map,
            'orientation': ori_map,
            'enhanced_real': enh_real,
            'enhanced_phase': enh_phase,
            'minutiae_orientation': mnt_o,
            'minutiae_x_offset': mnt_w,
            'minutiae_y_offset': mnt_h,
            'minutiae_score': mnt_s
        }

def get_fingernet_core(weights_path: str = DEFAULT_WEIGHTS_PATH, device: str = DEFAULT_DEVICE) -> FingerNet:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    logger.info(f"Selected device: {device}")
    logger.info("Loading FingerNet architecture...")
    fingernet_model = FingerNet()

    logger.info(f"Loading weights from: {weights_path}")
    fingernet_model.load_state_dict(torch.load(weights_path, map_location=device))
    fingernet_model.eval()
    fingernet_model.to(device)

    if device == "cpu":
        logger.info("Moving model to channels_last memory format...")
        fingernet_model.to(memory_format=torch.channels_last)

    logger.info("Model ready for inference.")

    return fingernet_model
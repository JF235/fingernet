import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from PIL import Image
import os
import kornia

DEFAULT_WEIGHTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "models", "released_version", "Model.pth")
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        t1 = time.time()
        x_norm = self.img_norm(x)
        t2 = time.time()
        features = self.feature_extractor(x_norm)
        t3 = time.time()
        ori_map, seg_map = self.ori_seg_head(features)
        t4 = time.time()
        enh_real, enh_phase, upsampled_ori_map = self.enhancement_module(x, ori_map)
        t5 = time.time()
        upsampled_seg = F.interpolate(nn.functional.softsign(seg_map), scale_factor=8, mode='nearest')
        upsampled_seg_out = F.interpolate(seg_map, scale_factor=8, mode='nearest')
        minutiae_input = torch.cat([enh_phase, upsampled_seg], dim=1)
        t6 = time.time()
        mnt_o, mnt_w, mnt_h, mnt_s = self.minutiae_head(minutiae_input, ori_map)
        t7 = time.time()

        print(f"ImgNorm: {t2 - t1:.4f}s, FeatExt: {t3 - t2:.4f}s, OriSeg: {t4 - t3:.4f}s, EnhMod: {t5 - t4:.4f}s, MinHead: {t7 - t6:.4f}s, Total= {t7 - t1:.4f}s")

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

class FingerNetWrapper(nn.Module):
    def __init__(self, model: FingerNet):
        super().__init__()
        self.fingernet = model

    def forward(self, x: torch.Tensor, minutiae_threshold: float = 0.5) -> dict[str, torch.Tensor]:
        
        padded_x = self.preprocess(x)
        
        with torch.no_grad():
            raw_outputs = self.fingernet(padded_x)
        
        final_outputs = self.postprocess(raw_outputs, threshold=minutiae_threshold)

        return final_outputs

    def time(self, x: torch.Tensor, minutiae_threshold: float = 0.5) -> dict[str, torch.Tensor]:
        
        t1 = time.time()
        padded_x = self.preprocess(x)
        t2 = time.time()

        with torch.no_grad():
            raw_outputs = self.fingernet.time(padded_x)

        t3 = time.time()
        final_outputs = self.postprocess(raw_outputs, threshold=minutiae_threshold)
        t4 = time.time()

        print(f"Preprocess: {t2 - t1:.4f}s, FingerNet: {t3 - t2:.4f}s, Postprocess: {t4 - t3:.4f}s, Total= {t4 - t1:.4f}s")
        return final_outputs

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        return F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

    def postprocess(self, outputs: dict, threshold: float) -> dict[str, torch.Tensor]:
        # 1. Binarização e limpeza da máscara de segmentação
        t1 = time.time()
        cleaned_mask = self._post_binarize_mask(outputs['segmentation'])
        cleaned_mask_up = self._post_binarize_mask(outputs['segmentation upsample'], upsample_factor=8)
        t2 = time.time()

        # 2. Detecção de minúcias (incluindo NMS)
        # O resultado é uma lista de tensores, um para cada imagem no lote.
        final_minutiae_list = self._post_detect_minutiae(outputs, cleaned_mask, threshold)
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

    def _post_binarize_mask(self, seg_map: torch.Tensor, upsample_factor: int = 1) -> torch.Tensor:
        """Binariza e limpa a máscara de segmentação usando Kornia."""
        seg_map_squeezed = seg_map.squeeze(1)
        binarized = torch.round(seg_map_squeezed)
        kernel = torch.ones(5 * upsample_factor, 5 * upsample_factor, device=seg_map.device)
        # Kornia espera um shape [B, C, H, W], por isso o unsqueeze/squeeze
        cleaned = kornia.morphology.opening(binarized.unsqueeze(1), kernel).squeeze(1)
        return cleaned

    def _post_detect_minutiae(self, outputs: dict, cleaned_mask: torch.Tensor, threshold: float) -> list:
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
            final_minutiae = self._post_nms(minutiae_raw)
            final_minutiae_list.append(final_minutiae)
            
        return final_minutiae_list

    def _post_nms(self, minutiae: torch.Tensor, dist_thresh: float = 16.0, angle_thresh: float = torch.pi/6) -> torch.Tensor:
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

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        padded_x = self.preprocess(x)
        with torch.no_grad():
            seg_map = self.fingernet.segment(padded_x)
        return seg_map[:, :, :x.shape[2], :x.shape[3]]
    
    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        padded_x = self.preprocess(x)
        with torch.no_grad():
            enh_real = self.fingernet.enhance(padded_x)
        return enh_real[:, :, :x.shape[2], :x.shape[3]]


def get_fingernet_core(weights_path: str = DEFAULT_WEIGHTS_PATH, device: str = DEFAULT_DEVICE, log: bool = True) -> FingerNet:
    """
    Gets the FingerNet model.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Arquivo de pesos não encontrado em: {weights_path}")

    # 1. Detectar dispositivo
    if log: print(f"[FingerNet] Dispositivo selecionado: {device}")

    # 2. Instanciar o modelo base
    if log: print("[FingerNet] Carregando arquitetura FingerNet...")
    fingernet_model = FingerNet()

    # 3. Carregar os pesos (state_dict)
    if log: print(f"[FingerNet] Carregando pesos de: {weights_path}")
    fingernet_model.load_state_dict(torch.load(weights_path, map_location=device))
    fingernet_model.eval()
    fingernet_model.to(device)

    if device == "cpu":
        print("[FingerNet] Movendo o modelo para o formato de memória channels_last...")
        fingernet_model.to(memory_format=torch.channels_last)

    if log: print("\n[FingerNet] Modelo pronto para inferência.")

    return fingernet_model

def get_fingernet(weights_path: str = DEFAULT_WEIGHTS_PATH, device: str = DEFAULT_DEVICE, log: bool = True) -> FingerNetWrapper:
    """
    Gets the 
    1. preloaded, 
    2. device specific and 
    3. eval mode 
    FingerNetWrapper.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Arquivo de pesos não encontrado em: {weights_path}")

    # 1. Detectar dispositivo
    if log: print(f"[FingerNet] Dispositivo selecionado: {device}")

    # 2. Instanciar o modelo base
    if log: print("[FingerNet] Carregando arquitetura FingerNet...")
    fingernet_model = FingerNet()

    # 3. Carregar os pesos (state_dict)
    if log: print(f"[FingerNet] Carregando pesos de: {weights_path}")
    fingernet_model.load_state_dict(torch.load(weights_path, map_location=device))
    fingernet_model.eval()

    # 5. Criar e mover o wrapper para o dispositivo
    if log: print("[FingerNet] Criando e movendo o wrapper para o dispositivo...")
    fnet_wrapper = FingerNetWrapper(model=fingernet_model).to(device)

    if device == "cpu":
        print("[FingerNet] Movendo o wrapper para o formato de memória channels_last...")
        fnet_wrapper.to(memory_format=torch.channels_last)

    if log: print("\n[FingerNet] Modelo pronto para inferência.")
    
    return fnet_wrapper
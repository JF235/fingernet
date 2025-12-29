"""
Script para converter o modelo FingerNet (core) para ONNX.

Este script oferece duas opções:
1. Converter o modelo completo (com extração de minúcias)
2. Converter apenas até a segmentação + enhancement (sem minúcias)

Uso:
    python convert_to_onnx.py --weights <caminho_pesos> --output <saida.onnx> [--no-minutiae]

Exemplo:
    # Modelo completo
    python convert_to_onnx.py --weights ../models/released_version/Model.pth --output fingernet_full.onnx
    
    # Sem extração de minúcias (apenas segmentação + enhancement)
    python convert_to_onnx.py --weights ../models/released_version/Model.pth --output fingernet_seg_enh.onnx --no-minutiae
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys

# Adiciona o diretório pai ao path para importar fingernet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fingernet.model import FingerNet, ImgNormalization, FeatureExtractor, OrientationSegmentationHead, EnhancementModule, MinutiaeHead


class FingerNetSegmentationEnhancement(nn.Module):
    """
    Versão do FingerNet que retorna apenas:
    - Segmentação (upsampled)
    - Imagem melhorada (real)
    - Campo de orientação (upsampled)
    
    Ideal para aplicações que não precisam da extração de minúcias.
    """
    def __init__(self, fingernet_full: FingerNet):
        super().__init__()
        self.img_norm = fingernet_full.img_norm
        self.feature_extractor = fingernet_full.feature_extractor
        self.ori_seg_head = fingernet_full.ori_seg_head
        self.enhancement_module = fingernet_full.enhancement_module

    def forward(self, x: torch.Tensor):
        # Pipeline até enhancement
        x_norm = self.img_norm(x)
        features = self.feature_extractor(x_norm)
        ori_map, seg_map = self.ori_seg_head(features)
        enh_real, enh_phase, upsampled_ori_map = self.enhancement_module(x, ori_map)
        upsampled_seg_out = F.interpolate(seg_map, scale_factor=8, mode='nearest')
        
        # Retorna apenas as saídas relevantes
        return {
            'segmentation': upsampled_seg_out,
            'enhanced_image': enh_real,
            'orientation': upsampled_ori_map
        }


class FingerNetNativeResolution(nn.Module):
    """
    Versão do FingerNet que retorna:
    - Segmentação nativa (1/8): (batch, 1, H/8, W/8)
    - Campo de orientação nativo (1/8): (batch, 90, H/8, W/8)
    - Imagem melhorada (full res): (batch, 1, H, W)
    
    O Enhancement Module é aplicado (para gerar a imagem melhorada),
    mas as máscaras de segmentação e orientação permanecem em resolução 1/8.
    
    Ideal para aplicações que precisam da imagem melhorada mas querem
    economizar memória/latência nas máscaras de segmentação e orientação.
    """
    def __init__(self, fingernet_full: FingerNet):
        super().__init__()
        self.img_norm = fingernet_full.img_norm
        self.feature_extractor = fingernet_full.feature_extractor
        self.ori_seg_head = fingernet_full.ori_seg_head
        self.enhancement_module = fingernet_full.enhancement_module

    def forward(self, x: torch.Tensor):
        # Pipeline até enhancement
        x_norm = self.img_norm(x)
        features = self.feature_extractor(x_norm)
        ori_map, seg_map = self.ori_seg_head(features)
        
        # Enhancement gera imagem melhorada em resolução completa
        enh_real, enh_phase, upsampled_ori_map = self.enhancement_module(x, ori_map)
        
        # Retorna: seg e ori em 1/8, enhanced em resolução completa
        return {
            'segmentation_native': seg_map,
            'orientation_native': ori_map,
            'enhanced_image': enh_real
        }


class FingerNetONNXWrapper(nn.Module):
    """
    Wrapper para exportação ONNX que retorna tensores em vez de dicionário.
    ONNX não suporta dicionários como saída, então retornamos múltiplos tensores.
    """
    def __init__(self, model, include_minutiae=True, native_resolution=False):
        super().__init__()
        self.model = model
        self.include_minutiae = include_minutiae
        self.native_resolution = native_resolution

    def forward(self, x: torch.Tensor):
        if self.include_minutiae:
            # Modelo completo retorna todos os outputs
            out = self.model(x)
            return (
                out['orientation upsample'],
                out['segmentation upsample'],
                out['enhanced_real'],
                out['minutiae_orientation'],
                out['minutiae_x_offset'],
                out['minutiae_y_offset'],
                out['minutiae_score']
            )
        elif self.native_resolution:
            # Modelo em resolução nativa (1/8) retorna seg + ori nativos + enhanced
            out = self.model(x)
            return (
                out['segmentation_native'],
                out['orientation_native'],
                out['enhanced_image']
            )
        else:
            # Modelo sem minúcias retorna apenas seg + enh + ori
            out = self.model(x)
            return (
                out['segmentation'],
                out['enhanced_image'],
                out['orientation']
            )


def convert_to_onnx(weights_path: str, output_path: str, include_minutiae: bool = True, 
                    native_resolution: bool = False, input_shape: tuple = (1, 1, 400, 400), 
                    opset_version: int = 17):
    """
    Converte o modelo FingerNet para ONNX.
    
    Args:
        weights_path: Caminho para os pesos do modelo (.pth)
        output_path: Caminho de saída para o arquivo ONNX
        include_minutiae: Se False, exporta apenas segmentação + enhancement (sem minúcias)
        native_resolution: Se True, exporta saídas em resolução nativa 1/8 (sem upsampling)
        input_shape: Shape da entrada (batch, channels, height, width)
        opset_version: Versão do opset ONNX (default 17)
    """
    print(f"Carregando modelo de: {weights_path}")
    
    # Carrega o modelo completo
    fingernet_full = FingerNet()
    fingernet_full.load_state_dict(torch.load(weights_path, map_location='cpu'))
    fingernet_full.eval()
    
    # Escolhe a versão do modelo
    if include_minutiae:
        print("Exportando modelo COMPLETO (com extração de minúcias)")
        model = fingernet_full
        output_names = [
            'orientation_upsample',
            'segmentation_upsample',
            'enhanced_real',
            'minutiae_orientation',
            'minutiae_x_offset',
            'minutiae_y_offset',
            'minutiae_score'
        ]
    elif native_resolution:
        print("Exportando modelo em RESOLUÇÃO NATIVA 1/8 (seg + ori em 1/8, enhanced em full res)")
        model = FingerNetNativeResolution(fingernet_full)
        output_names = [
            'segmentation_native',
            'orientation_native',
            'enhanced_image'
        ]
    else:
        print("Exportando modelo SEM minúcias (apenas segmentação + enhancement)")
        model = FingerNetSegmentationEnhancement(fingernet_full)
        output_names = [
            'segmentation',
            'enhanced_image',
            'orientation'
        ]
    
    # Wrapper para ONNX (converte dict para tuple)
    model_wrapper = FingerNetONNXWrapper(model, include_minutiae, native_resolution)
    
    # Cria input dummy
    dummy_input = torch.randn(*input_shape)
    
    # Testa o modelo antes de exportar
    print("Testando o modelo antes da exportação...")
    with torch.no_grad():
        _ = model_wrapper(dummy_input)
    print("✓ Modelo testado com sucesso")
    
    # Exporta para ONNX
    print(f"Exportando para: {output_path}")
    torch.onnx.export(
        model_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=output_names,
        dynamic_axes={
            'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
            **{name: {0: 'batch_size'} for name in output_names}
        }
    )
    
    print(f"✓ Modelo exportado com sucesso para: {output_path}")
    print(f"  - Opset version: {opset_version}")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Outputs: {', '.join(output_names)}")
    
    # Verifica o arquivo ONNX
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ Modelo ONNX validado com sucesso")
    except ImportError:
        print("⚠ Pacote 'onnx' não encontrado. Instale com: pip install onnx")
    except Exception as e:
        print(f"⚠ Erro ao validar modelo ONNX: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Converte o modelo FingerNet para ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Modelo completo (com minúcias)
  python convert_to_onnx.py --weights ../models/released_version/Model.pth --output fingernet_full.onnx
  
  # Sem minúcias (apenas segmentação + enhancement)
  python convert_to_onnx.py --weights ../models/released_version/Model.pth --output fingernet_seg_enh.onnx --no-minutiae
  
  # Especificando tamanho de entrada customizado
  python convert_to_onnx.py --weights ../models/released_version/Model.pth --output model.onnx --height 512 --width 512
        """
    )
    
    parser.add_argument('--weights', type=str, required=True,
                        help='Caminho para o arquivo de pesos (.pth)')
    parser.add_argument('--output', type=str, required=True,
                        help='Caminho de saída para o arquivo ONNX')
    parser.add_argument('--no-minutiae', action='store_true',
                        help='Exporta apenas segmentação + enhancement (sem extração de minúcias)')
    parser.add_argument('--native-resolution', action='store_true',
                        help='Exporta saídas em resolução nativa 1/8 (sem upsampling, apenas seg + ori)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size para a entrada (default: 1)')
    parser.add_argument('--height', type=int, default=400,
                        help='Altura da imagem de entrada (default: 400)')
    parser.add_argument('--width', type=int, default=400,
                        help='Largura da imagem de entrada (default: 400)')
    parser.add_argument('--opset', type=int, default=17,
                        help='Versão do opset ONNX (default: 17)')
    
    args = parser.parse_args()
    
    # Valida argumentos
    if not os.path.exists(args.weights):
        print(f"Erro: Arquivo de pesos não encontrado: {args.weights}")
        sys.exit(1)
    
    # Valida combinação de argumentos
    if args.native_resolution and not args.no_minutiae:
        print("Aviso: --native-resolution implica --no-minutiae. Ajustando automaticamente.")
        args.no_minutiae = True
    
    # Cria diretório de saída se não existir
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Converte
    input_shape = (args.batch_size, 1, args.height, args.width)
    convert_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        include_minutiae=not args.no_minutiae,
        native_resolution=args.native_resolution,
        input_shape=input_shape,
        opset_version=args.opset
    )


if __name__ == '__main__':
    main()

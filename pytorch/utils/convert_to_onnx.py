"""
Converte o modelo FingerNet (core) para ONNX, com eixos dinâmicos (batch/H/W).

O ONNX existe para alimentar o pipeline C++ (cpp/include/fingernet/onnx_model.hpp),
e esse consumidor nunca lê os 286 canais grossos de orientação/offset: só o argmax
deles. Então o argmax vai no grafo -- ver ARGMAX_OUTPUTS.

Uso:
    python convert_to_onnx.py --weights <pesos.pth> --output <saida.onnx>
"""

import torch
import torch.nn as nn
import argparse
import os
import sys

# Adiciona o diretório pai ao path para importar fingernet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fingernet.model import FingerNet

#: Shape só para o tracing -- os eixos batch/H/W são dinâmicos no grafo final.
TRACE_SHAPE = (1, 1, 400, 400)

#: Saídas reduzidas a argmax no grafo, e o nome do plano de índices de cada uma.
#: São as quatro cujo pós-processamento é "qual canal ganhou": orientação (90 bins),
#: orientação da minúcia (180) e os dois offsets sub-bloco (8 cada). Transferir os
#: canais e reduzir no host custava 4.4 MB e 0.84 ms por imagem de 512x512 -- 22% do
#: forward em TRT fp16. ONNX ArgMax usa select_last_index=0, o mesmo primeiro-máximo
#: que fnpost::argmax_channels, então os índices são idênticos aos do host.
ARGMAX_OUTPUTS = {
    'orientation': 'orientation_index',
    'minutiae_orientation': 'minutiae_orientation_index',
    'minutiae_x_offset': 'minutiae_x_index',
    'minutiae_y_offset': 'minutiae_y_index',
}

#: Nomes das saídas do grafo, na ordem de FingerNet.OUTPUTS.
OUTPUT_NAMES = [ARGMAX_OUTPUTS.get(n, n) for n in FingerNet.OUTPUTS]


class _GraphOutput(nn.Module):
    """ONNX não aceita dict como saída; achata para tuple e reduz ARGMAX_OUTPUTS."""

    def __init__(self, model: FingerNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return tuple(
            torch.argmax(out[name], dim=1).to(torch.int32) if name in ARGMAX_OUTPUTS
            else out[name]
            for name in FingerNet.OUTPUTS
        )


def convert_to_onnx(weights_path: str, output_path: str, opset_version: int = 17):
    """
    Args:
        weights_path: Caminho para os pesos do modelo (.pth)
        output_path: Caminho de saída para o arquivo ONNX
        opset_version: Versão do opset ONNX
    """
    print(f"Carregando modelo de: {weights_path}")
    model = FingerNet()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    output_names = list(OUTPUT_NAMES)
    dynamic_axes = {
        'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
        **{name: {0: 'batch_size'} for name in output_names}
    }

    # dynamo=False força o exportador legado (o exportador dynamo do torch 2.10
    # gera um grafo inválido p/ este modelo -- sort topológico/initializer dup).
    print(f"Exportando para: {output_path}")
    torch.onnx.export(
        _GraphOutput(model),
        torch.randn(*TRACE_SHAPE),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print(f"✓ Exportado: {', '.join(output_names)}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(output_path))
        print("✓ Modelo ONNX validado")
    except ImportError:
        print("⚠ Pacote 'onnx' não encontrado. Instale com: pip install onnx")
    except Exception as e:
        print(f"⚠ Erro ao validar modelo ONNX: {e}")


def main():
    parser = argparse.ArgumentParser(description="Converte o modelo FingerNet para ONNX")
    parser.add_argument('--weights', type=str, required=True,
                        help='Caminho para o arquivo de pesos (.pth)')
    parser.add_argument('--output', type=str, required=True,
                        help='Caminho de saída para o arquivo ONNX')
    parser.add_argument('--opset', type=int, default=17)

    args = parser.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"Erro: arquivo de pesos não encontrado: {args.weights}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    convert_to_onnx(args.weights, args.output, opset_version=args.opset)


if __name__ == '__main__':
    main()

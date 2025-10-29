import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from pathlib import Path
import glob

def plot_img(ax: plt.Axes, image: np.ndarray):
    """Plota a imagem em um determinado eixo."""
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_ori_field(ax: plt.Axes, orientation_field: np.ndarray, stride: int = 16):
    """
    Sobrepõe o campo de orientação (segmentos) em um determinado eixo.

    Args:
        ax: O eixo do Matplotlib para desenhar.
        orientation_field: O array 2D com os ângulos em radianos.
        stride: O espaçamento entre os segmentos de orientação.
    """
    height, width = orientation_field.shape
    # O comprimento do segmento é proporcional ao stride para uma boa visualização
    segment_length = stride * 0.45 
    
    for r in range(stride // 2, height, stride):
        for c in range(stride // 2, width, stride):
            angle = orientation_field[r, c]
            # Ignora pontos sem orientação definida (onde o ângulo é 0 no background)
            if angle != 0:
                dx = segment_length * np.cos(angle)
                dy = segment_length * np.sin(angle)
                # Desenha uma linha do ponto (c, r) na direção do ângulo
                ax.plot([c - dx, c + dx], [r - dy, r + dy], 'r-', linewidth=1)

def plot_mnt(ax: plt.Axes, minutiae: np.ndarray, r: int = 10):
    """
    Sobrepõe as minúcias (quadrados e ângulos) em um determinado eixo.

    Args:
        ax: O eixo do Matplotlib para desenhar.
        minutiae: Array (N, 4) com colunas [x, y, ângulo, score].
        r: O comprimento do segmento que indica o ângulo da minúcia.
    """
    # Plota quadrados vermelhos sem preenchimento nas posições (x, y)
    ax.plot(
        minutiae[:, 0], 
        minutiae[:, 1], 
        'rs',  # 'r' para vermelho, 's' para quadrado (square)
        fillstyle='none', 
        markersize=6, 
        markeredgewidth=1
    )
    
    # Desenha os segmentos de orientação para cada minúcia
    for x, y, angle, score in minutiae:
        ax.plot([x, x + r * np.cos(angle)], [y, y + r * np.sin(angle)], 'r-', linewidth=1.5)

def plot_raw_output(
        output: dict,
        orig_img: np.ndarray | None = None,
        figsize: tuple = (20, 6),
        stride: int = 16
):
    orientation_field = output['orientation_field'].squeeze()
    enhanced_image = output['enhanced_image'].squeeze()
    minutiae = output['minutiae']

    if orig_img is None:
        input_image = enhanced_image
    else:
        input_image = orig_img

    # Cria a figura e a grade de subplots 1x4
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # --- Subplot 1 (Primeira Coluna) ---
    ax1 = axes[0]
    plot_img(ax1, orientation_field)
    ax1.set_title("Campo de Orientação")

    # --- Subplot 2 (Segunda Coluna) ---
    ax2 = axes[1]
    plot_img(ax2, enhanced_image)
    ax2.set_title("Imagem Melhorada")

    # --- Subplot 3 (Terceira Coluna) ---
    ax3 = axes[2]
    plot_img(ax3, input_image)
    plot_ori_field(ax3, orientation_field, stride=stride)
    ax3.set_title(f"Campo de Orientação (Stride: {stride})")
    
    # --- Subplot 4 (Quarta Coluna) ---
    ax4 = axes[3]
    plot_img(ax4, input_image)
    plot_mnt(ax4, minutiae)
    ax4.set_title(f"Minúcias Detectadas ({len(minutiae)})")

    # Ajusta o layout para evitar sobreposição de títulos
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta para o suptitle 
    

def plot_output(
    result: dict,
    save_path: str | None = None,
    stride: int = 16,
    figsize: tuple = (20, 6)
):
    """
    Gera uma figura 2x2 com a visualização completa dos resultados da inferência.

    Args:
        result (dict): Um único dicionário da lista de resultados de `run_inference`.
                       Deve conter as chaves 'input_path', 'orientation_field', etc.
        save_path (str | None): Caminho para salvar a figura. Se None, a figura é exibida.
        stride (int): O stride para a visualização do campo de orientação.
    """
    try:
        # Carrega a imagem de entrada original para sobreposição
        input_image = np.array(Image.open(result['input_path']).convert('L'))
    except FileNotFoundError:
        print(f"Erro: Imagem de entrada não encontrada em {result['input_path']}")
        return

    # Extrai os dados do dicionário de resultados
    orientation_field = result['orientation_field'].squeeze()
    enhanced_image = result['enhanced_image'].squeeze()
    minutiae = result['minutiae'][0]

    # Cria a figura e a grade de subplots 1x4
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # --- Subplot 1 (Primeira Coluna) ---
    ax1 = axes[0]
    plot_img(ax1, orientation_field)
    ax1.set_title("Campo de Orientação")

    # --- Subplot 2 (Segunda Coluna) ---
    ax2 = axes[1]
    plot_img(ax2, enhanced_image)
    ax2.set_title("Imagem Melhorada")

    # --- Subplot 3 (Terceira Coluna) ---
    ax3 = axes[2]
    plot_img(ax3, input_image)
    plot_ori_field(ax3, orientation_field, stride=stride)
    ax3.set_title(f"Campo de Orientação (Stride: {stride})")
    
    # --- Subplot 4 (Quarta Coluna) ---
    ax4 = axes[3]
    plot_img(ax4, input_image)
    plot_mnt(ax4, minutiae)
    ax4.set_title(f"Minúcias Detectadas ({len(minutiae)})")

    # Define um título geral para a figura
    base_name = os.path.basename(result['input_path'])
    fig.suptitle(f"Resultados da FingerNet para: {base_name}", fontsize=16)

    # Ajusta o layout para evitar sobreposição de títulos
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta para o suptitle

    # Salva ou exibe a figura
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📈 Visualização salva em: {save_path}")
    else:
        plt.show()
    
    # Fecha a figura para liberar memória
    plt.close(fig)


def plot_from_output_folder(
    output_path: str, 
    image_filename: str, 
    save_path: str | None = None, 
    stride: int = 16,
    degrees: bool = False
):
    """
    Plota os resultados da inferência a partir da nova estrutura de pastas,
    reconstruindo os caminhos para uma imagem específica.

    Args:
        output_path (str): Caminho para a pasta principal de resultados (ex: 'output/').
        image_filename (str): Nome do arquivo da imagem original (ex: '101_1.png').
        save_path (str | None): Caminho para salvar a figura. Se None, exibe na tela.
        stride (int): Stride para visualização do campo de orientação.
    """
    print(f"INFO: Gerando visualização para '{image_filename}' a partir de '{output_path}'...")

    # Use apenas o basename do nome do ficheiro fornecido pelo usuário.
    # Isso permite que o usuário passe tanto '100_1.png' quanto 'enhanced/100_1.png'.
    image_basename = os.path.basename(image_filename)
    base_name = Path(image_basename).stem

    # --- Reconstrói os caminhos dos arquivos com base na nova estrutura ---
    # Candidate directory name patterns for each artifact group.
    enhanced_dirs = ['enh*']
    orientation_dirs = ['ori*', 'orientation*']
    minutiae_dirs = ['mnt*', 'minutiae*']

    def find_file_by_dir_patterns(base_dir: str, dir_patterns: list[str], filename: str):
        """Search for filename inside subdirectories of base_dir matching any of dir_patterns.

        Returns the first full path found or None.
        """
        # Try exact path relative to base_dir first
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            return candidate

        for pat in dir_patterns:
            glob_path = os.path.join(base_dir, pat)
            for match in glob.glob(glob_path):
                if os.path.isdir(match):
                    full = os.path.join(match, filename)
                    if os.path.exists(full):
                        return full
        return None

    enhanced_path = find_file_by_dir_patterns(output_path, enhanced_dirs, image_basename)
    orientation_path = find_file_by_dir_patterns(output_path, orientation_dirs, image_basename)
    minutiae_path = find_file_by_dir_patterns(output_path, minutiae_dirs, f"{base_name}.txt")

    # Verifica se todos os arquivos necessários existem
    for name, path in (('enhanced', enhanced_path), ('orientation', orientation_path), ('minutiae', minutiae_path)):
        if path is None:
            print(f"ERRO: Não foi possível localizar o arquivo {name} para '{image_basename}' dentro de '{output_path}'.")
            return
        if not os.path.exists(path):
            print(f"ERRO: Arquivo necessário não encontrado: {path}")
            return

    # Carrega os dados dos arquivos
    enhanced_image = np.array(Image.open(enhanced_path).convert('L'))
    orientation_img = np.array(Image.open(orientation_path))
    # orientation_img is stored in degrees in many pipelines; subtract 90 then convert
    orientation_field = np.deg2rad(orientation_img.astype(np.float32) - 90.0)

    minutiae = np.loadtxt(minutiae_path, delimiter=',', skiprows=1)
    if minutiae.ndim == 1 and minutiae.size > 0: # Garante que funcione para uma única minúcia
        minutiae = np.expand_dims(minutiae, 0)
    elif minutiae.size == 0: # Lida com o caso de nenhuma minúcia encontrada
        minutiae = np.empty((0, 4))


    # --- Cria a figura com 3 subplots (lógica de plotagem inalterada) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Imagem melhorada
    plot_img(axes[0], enhanced_image)
    axes[0].set_title("Imagem Melhorada")

    # 2. Imagem melhorada + campo de orientação
    plot_img(axes[1], enhanced_image)
    plot_ori_field(axes[1], orientation_field, stride=stride)
    axes[1].set_title(f"Campo de Orientação (Stride: {stride})")

    # 3. Imagem melhorada + minúcias
    plot_img(axes[2], enhanced_image)
    # Convert minutiae angle column to radians if necessary
    if degrees:
        minutiae[:, 2] = np.deg2rad(minutiae[:, 2])
    plot_mnt(axes[2], minutiae)
    axes[2].set_title(f"Minúcias Detectadas ({len(minutiae)})")

    # Título geral e salvamento
    fig.suptitle(f"Resultados FingerNet para: {image_filename}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        # Garante que o diretório de destino para a imagem de plotagem exista
        plt.savefig(save_path)
        print(f"📈 Visualização salva em: {save_path}")
    else:
        # plt.show() pode causar erros em ambientes sem GUI
        print("AVISO: save_path não fornecido. A plotagem não será exibida em ambientes sem GUI.")
        # plt.show() 
    
    plt.close(fig)

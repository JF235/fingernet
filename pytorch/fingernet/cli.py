import argparse
import json
import sys


def parse_gpus(gpus_str: str):
    """
    Parse GPU specification from string.
    
    Examples:
        "0" -> 0 (CPU)
        "1" -> 1 (single GPU)
        "2" -> 2 (2 GPUs: 0,1)
        "[0,1,2,3]" -> [0,1,2,3] (specific GPUs)
        
    Returns:
        None, int, or list of ints
    """
    if gpus_str.lower() == 'none' or gpus_str == '0':
        return 0
    
    try:
        # Try to parse as integer
        parsed_int = int(gpus_str)
    except ValueError:
        parsed_int = None
    if parsed_int is not None:
        if parsed_int < 0:
            raise ValueError("GPU integer must be >= 0")
        return parsed_int
    
    try:
        # Try to parse as JSON list
        parsed = json.loads(gpus_str)
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            if not parsed:
                raise ValueError("GPU list cannot be empty")
            if any(x < 0 for x in parsed):
                raise ValueError("GPU list must contain only non-negative integers")
            if len(set(parsed)) != len(parsed):
                raise ValueError("GPU list cannot contain duplicates")
            return parsed
        raise ValueError("GPU list must contain only integers")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid GPU specification: {gpus_str}")


def _to_cuda_visible_devices(gpus: int | list[int]) -> str:
    """
    Convert parsed GPU selection to CUDA_VISIBLE_DEVICES value.

    Returns:
        Comma-separated list of physical GPU IDs, or empty string for CPU mode.
    """
    if gpus == 0:
        return ""
    if isinstance(gpus, int):
        # `--gpus N` means "use first N GPUs": [0, 1, ..., N-1]
        return ",".join(str(i) for i in range(gpus))
    return ",".join(str(i) for i in gpus)


def infer_command(args):
    """Execute full inference (forward pass)."""
    import os

    gpus = parse_gpus(args.gpus)

    # CUDA_VISIBLE_DEVICES is set early in _early_set_cuda_visible_devices()
    from .api import run_inference
    
    print(f"\n{'='*70}")
    print("FingerNet - Full Inference")
    print(f"{'='*70}")
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"GPUs:        {gpus}")
    print(f"Batch Size:  {args.batch_size} per GPU")
    print(f"Workers:     {args.cores} per GPU")
    print(f"Recursive:   {args.recursive}")
    print(f"Threshold:   {args.threshold}")
    print(f"Compile:     {args.compile}")
    print(f"Max Dim:     {args.max_dim}")
    print(f"Strategy:    {args.strategy}")
    print(f"CPU Workers: {args.cpu_workers}")
    print(f"Quality Mask:{args.quality_mask}")
    print(f"Unmodulated: {args.unmodulated}")
    print(f"Full Extr.:  {args.full}")
    print(f"{'='*70}\n")
    
    run_inference(
        input_path=args.input,
        output_path=args.output,
        weights_path=args.weights,
        gpus=gpus,
        batch_size=args.batch_size,
        num_workers=args.cores,
        recursive=args.recursive,
        mnt_degrees=args.degrees,
        threshold=args.threshold,
        compile_model=args.compile,
        max_image_dim=args.max_dim,
        strategy=args.strategy,
        num_cpu_workers=args.cpu_workers,
        quality_mask=args.quality_mask,
        unmodulated=args.unmodulated,
        full=args.full,
    )


def forward_command(args):
    """Alias for infer_command."""
    infer_command(args)

def plot_command(args):
    """Generate visualization for processed results."""
    # Lazy import so plotting code isn't imported on `-h`
    from .plot import plot_from_output_folder

    print(f"\n{'='*70}")
    print("FingerNet - Plot Results")
    print(f"{'='*70}")
    print(f"Output Path: {args.output}")
    print(f"Image:       {args.image}")
    
    if args.save:
        print(f"Save To:     {args.save}")
    print(f"{'='*70}\n")
    
    plot_from_output_folder(
        output_path=args.output,
        image_filename=args.image,
        save_path=args.save,
        stride=args.stride,
        degrees=args.degrees,
        input_path=args.input_image,
    )


def _early_set_cuda_visible_devices():
    """Parse --gpus from sys.argv before any imports and set CUDA_VISIBLE_DEVICES.

    Must run before torch is imported (even indirectly) because the CUDA
    runtime reads the env var once at initialization and ignores later changes.
    """
    import os

    # Only inference commands use --gpus. For those commands, if --gpus is omitted,
    # default CLI behavior is equivalent to --gpus 1.
    inference_cmds = {'infer', 'forward', 'enhance', 'segment'}
    if len(sys.argv) < 2 or sys.argv[1] not in inference_cmds:
        return

    gpus_str = '1'
    for i, arg in enumerate(sys.argv):
        if arg == '--gpus' and i + 1 < len(sys.argv):
            gpus_str = sys.argv[i + 1]
            break

    gpus = parse_gpus(gpus_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = _to_cuda_visible_devices(gpus)


def main():
    """Main CLI entry point."""
    _early_set_cuda_visible_devices()

    parser = argparse.ArgumentParser(
        prog='fingernet',
        description='FingerNet - Advanced Fingerprint Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full inference on single GPU
  fingernet infer images/ output/ --gpus 1 --batch-size 8
  
  # Multi-GPU inference (4 GPUs: 0,1,2,3)
  fingernet forward images/ output/ --gpus 4 --batch-size 4 --recursive
  
  # Specific GPUs
  fingernet infer images/ output/ --gpus [2,3] --batch-size 8
  
  # Enhancement only
  fingernet enhance images/ output/ --gpus 2
  
  # Plot results
  fingernet plot output/ image.png --save viz.png
        """
    )

    # If user requested help at the top-level without specifying a subcommand,
    # construct an expanded help that includes options for all subcommands.
    # If a subcommand is present (e.g. `fingernet infer --help`), let
    # argparse handle that and show only the subcommand's help.
    subcommand_names = ('infer', 'forward', 'enhance', 'segment', 'plot')
    # Define a single shared function to add common inference arguments.
    def add_inference_args(sp):
        sp.add_argument('input', type=str, help='Input: image file, directory, or .txt list')
        sp.add_argument('output', type=str, help='Output directory for results')
        sp.add_argument('--gpus', type=str, default='1', help='GPU configuration: "0" (CPU), "1" (single GPU), "2" (2 GPUs), "[0,1,2]" (specific GPUs)')
        sp.add_argument('--weights', type=str, default=None, help='Path to model weights (.pth file). Default: use bundled weights')
        sp.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size per GPU (default: 4)')
        sp.add_argument('--cores', type=int, default=4, help='CPU cores for data loading per GPU (default: 4)')
        sp.add_argument('--recursive', '-r', action='store_true', default=True, help='Search for images recursively in directories (default: on)')
        sp.add_argument('--threshold', type=float, default=0.5, help='Minutia quality threshold 0–1 (default: 0.5)')
        sp.add_argument('--degrees', action='store_true', default=True, help='Save minutiae angles in degrees instead of radians (default: on)')
        sp.add_argument('--compile', action='store_true', help='Compile model with torch.compile for faster inference (experimental)')
        sp.add_argument('--max-dim', type=int, default=1024, help='Maximum dimension (H or W) for an image before resizing (default: 1024)')
        sp.add_argument('--strategy',  type=str, default='full_gpu', choices=['hybrid', 'full_gpu'], help="Execution strategy: 'hybrid' (GPU infer, CPU post-proc) 'full_gpu' (everything on GPU). (default: full_gpu)")
        sp.add_argument('--cpu-workers', type=int, default=4, help='Number of CPU threads for post-processing in hybrid mode and for saving results (default: 4)')
        sp.add_argument('--quality-mask', action='store_true', help='Export continuous quality mask (raw sigmoid segmentation before binarization)')
        sp.add_argument('--unmodulated', action='store_true', help='Export unmodulated orientation, enhanced, and minutiae (without segmentation mask)')
        sp.add_argument('--full', action='store_true', help='Full extraction: export all standard outputs plus quality mask and unmodulated versions')

    if any(h in sys.argv for h in ('-h', '--help')) and not any(cmd in sys.argv for cmd in subcommand_names):
        subparsers_temp = parser.add_subparsers(dest='command', required=False, help='Command to execute')

        for name in ('infer', 'forward', 'enhance', 'segment'):
            sp = subparsers_temp.add_parser(name)
            add_inference_args(sp)

        sp = subparsers_temp.add_parser('plot')
        sp.add_argument('output', type=str, help='Output directory containing results (e.g., output/)')
        sp.add_argument('image', type=str, help='Image filename to visualize (e.g., 101_1.png)')
        sp.add_argument('--save', type=str, default=None, help='Path to save visualization (default: show in window)')
        sp.add_argument('--stride', type=int, default=16, help='Stride for orientation field visualization (default: 16)')

        # Print combined help and exit immediately
        print(parser.format_help())
        print('\nSUBCOMMANDS:\n')
        for name, sp in subparsers_temp.choices.items():
            print(f"== {name} ==")
            print(sp.format_help())
        return

    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to execute')
    
    # Reuse shared `add_inference_args` defined above.
    
    # --- 'infer' command (full inference) ---
    infer_parser = subparsers.add_parser(
        'infer',
        help='Run full inference (all outputs)',
        description='Execute complete FingerNet inference pipeline'
    )
    add_inference_args(infer_parser)
    infer_parser.set_defaults(func=infer_command)
    
    # --- 'forward' command (alias for infer) ---
    forward_parser = subparsers.add_parser(
        'forward',
        help='Run full inference (alias for infer)',
        description='Execute complete FingerNet inference pipeline (alias for infer)'
    )
    add_inference_args(forward_parser)
    forward_parser.set_defaults(func=forward_command)
    
    # --- 'plot' command ---
    plot_parser = subparsers.add_parser(
        'plot',
        help='Visualize inference results',
        description='Generate visualization from saved results'
    )
    plot_parser.add_argument(
        'output', type=str,
        help='Output directory containing results (e.g., output/)'
    )
    plot_parser.add_argument(
        'image', type=str,
        help='Image filename to visualize (e.g., 101_1.png)'
    )
    plot_parser.add_argument(
        '--save', type=str, default=None,
        help='Path to save visualization (default: show in window)'
    )
    plot_parser.add_argument(
        '--stride', type=int, default=16,
        help='Stride for orientation field visualization (default: 16)'
    )
    plot_parser.add_argument(
        '--degrees', action='store_true',
        help='Interpret stored orientation/minutiae angles as degrees (convert to radians before plotting)'
    )
    plot_parser.add_argument(
        '--input-image', type=str, default=None,
        help='Path to original input image (default: use enhanced image)'
    )
    plot_parser.set_defaults(func=plot_command)
    
    # Parse and execute
    args = parser.parse_args()
    
    # Set default weights path if not provided
    if hasattr(args, 'weights') and args.weights is None:
        from .model import DEFAULT_WEIGHTS_PATH
        args.weights = DEFAULT_WEIGHTS_PATH
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

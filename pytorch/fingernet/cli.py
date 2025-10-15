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
        return int(gpus_str)
    except ValueError:
        pass
    
    try:
        # Try to parse as JSON list
        parsed = json.loads(gpus_str)
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            return parsed
        raise ValueError("GPU list must contain only integers")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid GPU specification: {gpus_str}")


def infer_command(args):
    """Execute full inference (forward pass)."""
    # Lazy import to keep `fingernet -h` fast
    from .api import run_inference

    gpus = parse_gpus(args.gpus)
    
    print(f"\n{'='*70}")
    print("FingerNet - Full Inference")
    print(f"{'='*70}")
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"GPUs:        {gpus}")
    print(f"Batch Size:  {args.batch_size} per GPU")
    print(f"Workers:     {args.cores} per GPU")
    print(f"Recursive:   {args.recursive}")
    print(f"Compile:     {args.compile}")
    print(f"Max Dim:     {args.max_dim}")
    print(f"Strategy:    {args.strategy}")
    print(f"CPU Workers: {args.cpu_workers}")
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
        compile_model=args.compile,
        max_image_dim=args.max_dim,
        strategy=args.strategy,
        num_cpu_workers=args.cpu_workers,
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
        degrees=args.degrees
    )


def main():
    """Main CLI entry point."""
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
        sp.add_argument('--recursive', '-r', action='store_true', help='Search for images recursively in directories')
        sp.add_argument('--degrees', action='store_true', help='Save minutiae angles in degrees instead of radians')
        sp.add_argument('--compile', action='store_true', help='Compile model with torch.compile for faster inference (experimental)')
        sp.add_argument('--max-dim', type=int, default=1024, help='Maximum dimension (H or W) for an image before resizing (default: 1024)')
        sp.add_argument('--strategy',  type=str, default='full_gpu', choices=['hybrid', 'full_gpu'], help="Execution strategy: 'hybrid' (GPU infer, CPU post-proc) 'full_gpu' (everything on GPU). (default: full_gpu)")
        sp.add_argument('--cpu-workers', type=int, default=4, help='Number of CPU threads for post-processing in hybrid mode and for saving results (default: 4)')

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

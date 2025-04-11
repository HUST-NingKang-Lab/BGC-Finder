import sys
import os
from model_utils import get_CLI_parser

def main():
    print('Start runnning CoreFinder...')
    parser = get_CLI_parser()
    args = parser.parse_args()
    
    if args.mode == 'annotate':
        from annotate import annotate
        cfg = {
            'input': args.input, 
            'output': args.output, 
            'model': args.model
        }
        annotate(cfg, args)
        
    elif args.mode == 'pipeline':
        from prophet2fasta import prophet2fasta
        cfg = {
            'prophet': args.prophet,
            'original_fasta': args.original_fasta,
            'generated_fasta': args.generated_fasta
        }
        prophet2fasta(cfg, args)
        
        from annotate import annotate
        cfg = {
            'input': args.generated_fasta, 
            'output': args.output, 
            'model': args.model
        }
        annotate(cfg, args)
        
    else:
        print('Invalid mode. Please choose either "annotate" or "pipeline".')
        sys.exit(1)
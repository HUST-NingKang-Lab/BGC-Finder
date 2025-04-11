import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from Bio import SeqIO
from model_utils import extract_bgcs, write_fasta_files



def prophet2fasta(cfg, args):
    
    """
    Main function to process Prophet results and generate fasta files for BGCs.
    
    Parameters:
    prophet_res (pd.DataFrame): DataFrame containing Prophet results.
    fasta_dir (str): Directory containing the original fasta files.
    output_dir (str): Directory to save the generated fasta files.
    """
    
    prophet_res_path = None
    
    for file in os.listdir(cfg['prophet']):
        if file.endswith('_classified.csv'):
            prophet_res_path = os.path.join(cfg['prophet'], file)
            break
    
    if prophet_res_path is None:
        raise FileNotFoundError("No classified Prophet results found in the BGC-Prophet output directory.")
    
    prophet_res = pd.read_csv(prophet_res_path, index_col=0)

    output_dir = cfg['generated_fasta']
    fasta_dir = cfg['original_fasta']
    
    os.makedirs(output_dir, exist_ok=True)

    # Filter for BGCs
    prophet_res = prophet_res[prophet_res['isBGC'] == 'Yes']
    prophet_res['TDsentence'] = prophet_res['TDsentence'].str.split(' ')
    prophet_res['TDlabels'] = prophet_res['TDlabels'].str.split(' ').apply(lambda x: list(map(int, x)))
    prophet_res['genome'] = prophet_res.index.str.rsplit('_', n=1).str[0]

    # Extract BGCs
    bgcs = {}
    print('Extracting BGCs...')
    for genome, TDsentence, TDlabels in tqdm(zip(prophet_res['genome'], prophet_res['TDsentence'], prophet_res['TDlabels']), total=len(prophet_res)):
        if genome not in bgcs:
            bgcs[genome] = extract_bgcs(TDsentence, TDlabels)
        else:
            bgcs[genome].extend(extract_bgcs(TDsentence, TDlabels))
        
    count = sum(len(bgcs[genome]) for genome in bgcs)
    
    # Write sequences to fasta files
    write_fasta_files(output_dir, fasta_dir, bgcs)
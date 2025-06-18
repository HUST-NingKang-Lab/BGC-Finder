import os
import argparse
from Bio import SeqIO
import numpy as np
import pandas as pd
import pickle as pk
import pkg_resources
from tqdm import tqdm
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import EsmModel, EsmTokenizer, RobertaForTokenClassification



class Seq4Transformer(Dataset):
    def __init__(self, seqs, tokenizer, max_len=1024):
        self.seqs = seqs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        toks = self.tokenizer(self.seqs[idx], truncation=True, padding='max_length', return_tensors='pt', max_length=self.max_len)
        return {'input_ids': toks['input_ids'].squeeze(),
                'attention_mask': toks['attention_mask'].squeeze()}
        
def get_embeddings(embedder, dataset, embed_device):
    """
    Get the embeddings for the sequences in the dataset.
    
    Args:
        embedder (EsmModel): ESM model for generating embeddings.
        tokenizer (EsmTokenizer): Tokenizer for the ESM model.
        dataset (Dataset): Dataset containing the sequences.
        
    Returns:
        np.ndarray: Array of embeddings for the sequences.
    """
    reps = []

    for seq in dataset:
        res = embedder(seq['input_ids'].unsqueeze(0).to(embed_device),
                    seq['attention_mask'].unsqueeze(0).to(embed_device))
        token_representations = res['hidden_states'][33]
        # get mean representation of the sequence along attention mask
        # remove the bos, eos and padding tokens
        len = seq['attention_mask'].sum().item()
        token_representations = token_representations[:, 1:len-1, :]
        mean_representations = token_representations.mean(dim=1).squeeze()  # (1, 1280)
        reps.append(mean_representations.cpu().detach())
        
        del res, token_representations, mean_representations
        torch.cuda.empty_cache()
        
    reps = torch.stack(reps)
    cls_embed = torch.zeros(1, reps.shape[-1], dtype=torch.float32)
    reps = torch.cat([cls_embed, reps], dim=0)
    token_type_ids = torch.ones(reps.shape[0], dtype=torch.long)
    token_type_ids[0] = 0
    return reps, token_type_ids


def find_pkg_resource(path):
    
    if pkg_resources.resource_exists('CoreFinder', path):
        return pkg_resources.resource_filename('CoreFinder', path)
    else:
        raise FileNotFoundError("Resource {} not found.".format(path))
    

def get_CLI_parser():
    
    modes = ['annotate', 'pipeline']
    
    parser = argparse.ArgumentParser(
        description=('CoreFinder: A tool for deciphering biosynthetic gene clusters (BGCs).'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('mode', type=str, default='annotate_genbank', 
                        choices=modes,
                        help='Mode of operation. Available modes: {}'.format(modes))
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input file. Required.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output folder. Required.')
    parser.add_argument('-m', '--model', type=str, default='model/',
                        help='Model folder path')
    
    # --------------------------------------------------
    
    pipeline = parser.add_argument_group(
        title='pipeline',
        description='Convert outputs from BGC-Prophet, then run CoreFinder.'
                    'For BGC detection and deciphering, please run BGC-Prophet first.'
                    'and use the pipeline mode.'
                    'BGC-Prophet link: https://github.com/HUST-NingKang-Lab/BGC-Prophet')
    pipeline.add_argument('-p', '--prophet', type=str, required=True,
                            help='BGC-Prophet output folder. Required.')
    pipeline.add_argument('-of', '--original_fasta', type=str, required=True,
                            help='Fasta file folder of original genomes. Required.')
    pipeline.add_argument('-gf', '--generated_fasta', type=str, required=True,
                            help='Fasta file folder to save results from BGC-Prophet. Required.')
    
    return parser


def get_gene_label(logit):
    logit = np.array(logit)
    if logit.argmax() < 7:
        gene_label = 'other'
    else:
        logit = logit[-5:]
        logit = np.exp(logit) / np.sum(np.exp(logit))
        label_dict = ['biosynthetic', 'biosynthetic-additional', 
                        'other', 'regulatory', 'transport']
        gene_label = label_dict[logit.argmax()]
    return gene_label


def extract_bgcs(TDsentence, TDlabels):
    """
    Extracts BGCs (Biosynthetic Gene Clusters) from the given TDsentence and TDlabels.
    
    Parameters:
    TDsentence (list): List of gene names.
    TDlabels (list): List of labels indicating BGC presence (1 for BGC, 0 for non-BGC).
    
    Returns:
    list: List of BGCs, where each BGC is a sublist of genes.
    """
    bgcs = []
    start = np.where(np.diff(TDlabels) == 1)[0] + 1 
    end = np.where(np.diff(TDlabels) == -1)[0] + 1

    if TDlabels[0] == 1:    # If the first BGC starts at the beginning of the sequence
        start = np.append(0, start)
    if TDlabels[-1] == 1:   # If the last BGC ends at the end of the sequence
        end = np.append(end, len(TDlabels))

    for s, e in zip(start, end):
        bgcs.append(TDsentence[s:e])

    return bgcs

def write_fasta_files(output_dir, fasta_dir, bgcs):
    """
    Writes the BGC gene sequences to fasta files.
    
    Parameters:
    output_dir (str): Directory to save the fasta files.
    fasta_dir (str): Directory containing the original fasta files.
    bgcs (dict): Dictionary of BGCs with genome names as keys and lists of BGC genes as values.
    sequences (dict): Dictionary with gene names as keys and sequences as values.
    """
    print('Writing fasta files...')
    for genome, bgc_genes in tqdm(bgcs.items()):
        sequences = {}
        for record in SeqIO.parse(os.path.join(fasta_dir, f'{genome}.fasta'), 'fasta'):
            sequences[record.id] = record.seq
        for i, bgc in enumerate(bgc_genes):
            output_path = os.path.join(output_dir, f'{genome}_cluster{i+1}.fasta')
            with open(output_path, 'w') as f:
                for gene in bgc:
                    f.write(f'>{gene}\n{sequences[gene]}\n')


    

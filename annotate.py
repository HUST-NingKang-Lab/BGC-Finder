import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from pickle import load, dump
from transformers import EsmModel, EsmTokenizer, RobertaForTokenClassification
from corefinder import get_embeddings, Seq4Transformer
from model_utils import get_gene_label
from Bio.Seq import Seq
from Bio import SeqIO
import os


def annotate(cfg, args):
    
    """
    Annotate a GenBank file with the CoreFinder model.
    
    Args:
        cfg (dict): Configuration dictionary.
        args (argparse.Namespace): Command line arguments.
        
    Returns:
        {[gene_product], [gene_function], [gene_name]}: Dictionary containing the annotations.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model and tokenizer
    embedder_path = os.path.join(cfg['model'], 'esm_model')
    embedder = EsmModel.from_pretrained(embedder_path).to(device)
    tokenizer = EsmTokenizer.from_pretrained(embedder_path)
    
    
    input_file_path = cfg['input']
    output_folder_path = cfg['output']
    
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Load the sequences from the GenBank file
    records = SeqIO.parse(args.input, 'genbank')
    
    # read genbank / fasta file
    seqs = []
    names = []
    
    if input_file_path.endswith('.gb') or input_file_path.endswith('.gbk'):
        records = SeqIO.parse(args.input, 'genbank')
        for record in records:
            for feature in record.features:
                if feature.type == 'CDS':
                    name = feature.qualifiers['locus_tag'][0]
                    seq = feature.qualifiers['translation'][0]
                    seqs.append(seq)
                    names.append(name)
    
    elif input_file_path.endswith('.fasta') or input_file_path.endswith('.fa'):
        records = SeqIO.parse(args.input, 'fasta')
        for record in records:
            name = record.id
            seq = str(record.seq)
            seqs.append(seq)
            names.append(name)
            
    else:
        raise ValueError("Input file must be in GenBank (.gb/.gbk) or FASTA (.fasta/.fa) format.")
    
    dataset = Seq4Transformer(seqs, tokenizer)
    
    # Get the embeddings for the sequences
    reps, token_type_ids = get_embeddings(embedder, dataset, device)
    
    # Load the model for classification
    corefinder_path = os.path.join(cfg['model'], 'corefinder_model')
    model = RobertaForTokenClassification.from_pretrained(corefinder_path).to(device)
    
    outputs = model(inputs_embeds=reps.unsqueeze(0),
            token_type_ids=token_type_ids.unsqueeze(0)
            ).logits.squeeze().detach().cpu().numpy()
    
    logits = outputs.logits.squeeze()
    preds = {}
    for i in range(len(logits)):
        pred = logits[i]
        preds['gene_function'] = []
        
        if i == 0:
            prod = np.array(pred)[:7]
            prod_dict = ['Alkaloid', 'NRP', 'Other', 'Polyketide', 'RiPP', 'Saccharide','Terpene']
            prod = np.exp(prod) / np.sum(np.exp(prod))
            prod_label = prod_dict[prod.argmax()]
            preds['product'] = prod_label
            
        else:
            gene_label = get_gene_label(pred)
            preds['gene_function'].append(gene_label)
    
    preds['gene_name'] = names
    # print(preds)
    # Save the predictions to a pickle file
    output_file = os.path.join(output_folder_path, '{}_annotations.pkl'.format(os.path.basename(input_file_path)))
    
    with open(output_file, 'wb') as f:
        dump(preds, f)
    print(f'Annotations saved to {output_file}')
    
    print(preds)

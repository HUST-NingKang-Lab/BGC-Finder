import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pickle import load, dump
import time
from transformers import EsmModel, EsmTokenizer, RobertaForTokenClassification
from Bio import SeqIO
import os


class Seq4Transformer(Dataset):
    def __init__(self, seqs, tokenizer, max_len=2048):
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


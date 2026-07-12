import torch.nn as nn
import torch
from typing import Dict

class TokenEmbedding(nn.Module):
    """
    Token embedding layer that combines bin embeddings with positional embeddings.
    
    This module creates embeddings for numerical and categorical sequences by:
    1. Converting bin IDs to embeddings
    2. Adding positional information
    3. Adding CLS token
    
    Args:
        encoding_info (Dict[str, Dict[str, int]]): Nested dictionary containing encoding
                                                   information for each variable/column
        embedding_dim (int): Dimension of the embedding vectors
        dropout (float): Dropout rate of the embedding layer
        mode (str): Mode for concatenating bin and positional embeddings. Default: 'add'
    """
    
    def __init__(self,
                 encoding_info: Dict[str, Dict[str, int]],
                 embedding_dim: int=256,
                 dropout: float=0.1,
                 mode: str='add') -> None:
        super(TokenEmbedding, self).__init__()
        self.encoding_info = encoding_info
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.mode = mode
        
        # Initialize embedding layers
        num_encoding_info = {i: v['num_bins'] for i, (_, v) in enumerate(encoding_info.items()) if 'num_bins' in v.keys()}
        self.num_var_ids = list(num_encoding_info.keys())
        if len(self.num_var_ids) > 0:
            self.num_embedding = BinEmbedding(max_len=max(num_encoding_info.values()),
                                                  embedding_dim=embedding_dim,
                                                  mask_idx=0)
        else:
            self.num_embedding = None
        
        cat_encoding_info = {i: v['num_categories'] for i, (_, v) in enumerate(encoding_info.items()) if 'num_categories' in v.keys()}
        self.cat_var_ids = list(cat_encoding_info.keys())
        if len(self.cat_var_ids) > 0:
            self.cat_embedding = BinEmbedding(max_len=max(cat_encoding_info.values()),
                                                  embedding_dim=embedding_dim,
                                                  mask_idx=0)
        else:
            self.cat_embedding = None
        self.register_buffer('sorting_key', torch.argsort(torch.tensor(self.num_var_ids + self.cat_var_ids)))

        # Pre-register [CLS] token as a buffer for efficiency
        self.cls_embedding = nn.Embedding(1, embedding_dim)
        self.register_buffer('cls_token', torch.zeros(1, 1, dtype=torch.long))
        
        # Initialize positional embedding layer
        self.positional_embedding = PositionalEmbedding(max_len=len(self.num_var_ids + self.cat_var_ids),
                                                        embedding_dim=embedding_dim)
        
        # Initialize projection layer if mode is concat
        if mode == 'add':
            pass
        elif mode == 'concat':
            self.proj = nn.Linear(2 * embedding_dim, embedding_dim)
        else:
            raise ValueError("mode must be add or concat")
        
        # Initialize layer norm and dropout
        self.embedding_layernorm = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
    def forward(self, bin_ids: torch.Tensor) -> torch.Tensor:
        batch_size = bin_ids.size(0)
        
        # Create CLS tokens for the batch
        cls_token = self.cls_token.expand(batch_size, -1)
        cls_embedded = self.cls_embedding(cls_token)
        
        # Embedding layer: bin_ids -> embeddings
        num_embedded = self.num_embedding(bin_ids[:, self.num_var_ids]) if len(self.num_var_ids) > 0 else None
        cat_embedded = self.cat_embedding(bin_ids[:, self.cat_var_ids]) if len(self.cat_var_ids) > 0 else None
        bin_embeddings = torch.cat([e for e in [num_embedded, cat_embedded] if e is not None], dim=1)[:, self.sorting_key]
        
        # Get positional embeddings
        positional_embeddings = self.positional_embedding(bin_ids)       
        
        # Combine bin and positional embeddings
        if self.mode == 'add':
            bin_embeddings = bin_embeddings + positional_embeddings
        elif self.mode == 'concat':
            bin_embeddings = torch.cat([bin_embeddings, positional_embeddings.expand(bin_ids.size(0), -1, -1)], dim=-1)
            bin_embeddings = self.proj(bin_embeddings) 
        
        # Prepend CLS token
        embeddings = torch.cat([cls_embedded, bin_embeddings], dim=1)
        
        # Apply layer norm and dropout
        embeddings = self.embedding_layernorm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        return embeddings



class BinEmbedding(nn.Module):
    """
    Numerical and categorical embedding layer.
    
    This module creates embeddings for bin indices.
    
    Args:
        max_len (int): Maximum length of the input sequence (number of bin indices)
        embedding_dim (int): Dimension of the embedding vectors
        mask_idx (int, optional): Index used for masking. 
    """
    
    def __init__(self, 
                 max_len: int,
                 embedding_dim: int,
                 mask_idx: int=None) -> None:
        super(BinEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.mask_idx = mask_idx
        
        if mask_idx is not None:
            max_len += 1
            
        self.embedding = nn.Embedding(max_len, embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # The standard deviation of the initial weights is set to be larger than 
        # the standard LLM embeddings to activate the fused-type regularization.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        
    def forward(self, bin_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BinEmbedding layer.
        
        Args:
            bin_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing bin indices
        
        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding(bin_ids)


        
class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding layer that adds position information to sequences.
    
    This module creates position-dependent embeddings that are added to input embeddings
    to provide the model with information about the position (column) of each element in the sequence.
    
    Args:
        max_len (int): Maximum sequence length that can be processed (number of columns)
        embedding_dim (int): Dimension of the embedding vectors
    """
    
    def __init__(self,
                 max_len: int,
                 embedding_dim: int):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_len, embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # The standard deviation of the initial weights is set to be less than 
        # that of the bin embeddings to prevent dominance of positional information.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalEmbedding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
                             Used only to determine sequence length and device
        
        Returns:
            torch.Tensor: Positional embeddings of shape (batch_size, sequence_length, embedding_dim)
        """
        seq_len = x.size(1)
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        return self.embedding(positions)


    
if __name__ == '__main__':
    embedding = BinEmbedding(10, 3, 16)
    x = torch.randint(10, (256, 3))
    embedded = embedding(x)
    print(embedded.shape)
    
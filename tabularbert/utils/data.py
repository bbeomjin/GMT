import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List
from torch.utils.data import Dataset
from .type import ArrayLike



def get_n_categories(x: ArrayLike) -> int:
    return len(np.unique(x))



class DiscretizeBase:
    """
    Base class for the discretization classes, ['QunatileDiscretize', 'UniformDiscretize'].
    encoding_info: Dictionary mapping variable names to their encoding specifications.
                   Format: {variable_name: {encoding_type: count}}, e.g. {'var1': {'num_bins': 10}, 'var2': {'num_categories': 5}}
    """
    
    def __init__(self, 
                 num_bins: int=10,
                 encoding_info: Dict[str, Dict[str, int]]=None
                 ) -> None:
       
       # Default number of bins
       self.num_bins = int(num_bins)
       
       # Initialize encoding_info
       if encoding_info is not None:
           categorical_vars = [k for k, v in encoding_info.items() if v == 'categorical']
           for c in categorical_vars:
               encoding_info[c] = {'num_categories': None}
       
       # Correct encoding format
       if encoding_info is not None:
           for k, v in encoding_info.items():
               if 'num_bins' in v.keys() and (isinstance(v['num_bins'], int) or isinstance(v['num_bins'], float)):
                   encoding_info[k]['num_bins'] = int(v['num_bins'])
               if 'num_categories' in v.keys() and (isinstance(v['num_categories'], int) or isinstance(v['num_categories'], float)):
                   encoding_info[k]['num_categories'] = int(v['num_categories'])
       
       # Specific binning information
       self.encoding_info = encoding_info
       
    def _fit(self,
             x: ArrayLike,
             num_bins: int,
             ) -> List[float]:
        raise NotImplementedError()
    
    def fit(self,
            x: ArrayLike
            ) -> None:
        
        if isinstance(x, pd.DataFrame):
            self.columns = list(x.columns)
            categorical = x.dtypes.isin(['object', 'category']).to_list()
            categorical_vars = [c for c, v in zip(self.columns, categorical) if v]
            if len(categorical_vars) > 0:
                if self.encoding_info is None:
                    self.encoding_info = {v: {'num_categories': None} for v in categorical_vars}
                else:
                    self.encoding_info.update({v: {'num_categories': None} for v in categorical_vars if v not in self.encoding_info.keys()})
            x = x.values
        else:
            self.columns = [j for j in range(x.shape[1])]
        
        # Set the default number of bins
        encoding_info = {k: {'num_bins': self.num_bins} for k in self.columns}
        
        if self.encoding_info is not None:
            vars = list(self.encoding_info.keys())
            if all([k in encoding_info.keys() for k in vars]) is not True:
                raise ValueError(
                    f"Column(s) specified in 'encoding_info' are not found in the input data: {vars}"
                )
            for v in vars:
                encoding_info[v] = self.encoding_info[v]
        
        for i, (k, v) in enumerate(encoding_info.items()):
            if 'num_categories' in v.keys() and v['num_categories'] is None:
                v['num_categories'] = get_n_categories(x=x[:, i])
        
        self.encoding_info = encoding_info
        
        # Getting cut-off values for binning
        bins = {}
        category_maps = {}
        for j, (k, v) in enumerate(encoding_info.items()):
            if 'num_bins' in v.keys():
                xx = x[:, j]
                if np.issubdtype(xx.dtype, np.number) is not True:
                    xx = xx.astype(np.float64)
                bins[k] = self._fit(xx, num_bins=v['num_bins'])
            elif 'num_categories' in v.keys():
                xx = x[:, j]
                categories = np.sort(np.unique(xx))
                category_maps[k] = {cat: idx + 1 for idx, cat in enumerate(categories)}
            else:
                bins[k] = None        
        
        self.bins = bins
        self.category_maps = category_maps
        
    def _discretize(self, 
                    x: ArrayLike,
                    bins: List[float] | ArrayLike,
                    ) -> ArrayLike:

        ids = np.digitize(x, bins=bins, right=False)
        # Bin index starts with 1
        return ids.astype(int) + 1
    
    def discretize(self, 
                   x: ArrayLike):
        
        if isinstance(x, pd.DataFrame):
            x = x.values
        
        if len(self.encoding_info) != x.shape[1]:
            raise ValueError(
                "The number of columns in the data to be discretized does not match the number of columns in the fitted data."
            )
        
        bin_ids_list = list()
        for j, (k, v) in enumerate(self.encoding_info.items()):
            if 'num_bins' in v.keys():
                xx = x[:, j]
                if np.issubdtype(xx.dtype, np.number) is not True:
                    xx = xx.astype(np.float64)
                bin_ids = self._discretize(x=xx, 
                                           bins=self.bins[k])
            elif 'num_categories' in v.keys():
                codes = self.category_maps[k]
                bin_ids = np.array([codes[val] for val in x[:, j]])
            else:
                raise ValueError(
                    "The encoding information is not valid."
                )
                
            bin_ids_list.append(bin_ids)

        return np.stack(bin_ids_list, axis = 1)



class QuantileDiscretize(DiscretizeBase):
    def __init__(self, 
                 num_bins: int = 10,
                 encoding_info: Dict[str, Dict[str, int]] = None
                 ) -> None:
        
        super(QuantileDiscretize, self).__init__(
            num_bins = num_bins,
            encoding_info = encoding_info
        )
        
    def _fit(self, 
             x: ArrayLike,
             num_bins: int,
             ) -> ArrayLike:
        bins = np.quantile(x, np.linspace(0, 1, num_bins + 1))
        bins[-1] = np.inf
        return bins[1:]



class UniformDiscretize(DiscretizeBase):
    def __init__(self, 
                 num_bins: int = 10,
                 encoding_info: Dict[str, Dict[str, int]] = None
                 ) -> None:
        
        super(UniformDiscretize, self).__init__(
            num_bins = num_bins,
            encoding_info = encoding_info
        )
        
    def _fit(self, 
             x: ArrayLike,
             num_bins: int,
             ) -> ArrayLike:
        bins = np.linspace(np.min(x), np.max(x), num_bins + 1)
        bins[-1] = np.inf
        return bins[1:]   



class SSLDataset(Dataset):
    """
    Dataset class for TabularBERT masked language modeling pretraining.
    
    This dataset handles the masking strategy for self-supervised learning on
    tabular data that has been discretized into bins. It applies three types of
    token transformations: masking, random replacement, and keeping unchanged.
    
    Args:
        x (ArrayLike): Original tabular data
        bin_ids (ArrayLike): Discretized tabular data as bin indices
        encoding_info (Dict[str, int]): Mapping of feature names to number of bins
        mask_token_id (int): Token ID used for masking. Default: 0
        mask_token_prob (float): Probability of masking tokens. Default: 0.15
        random_token_prob (float): Probability of random token replacement. Default: 0.1
        unchanged_token_prob (float): Probability of keeping tokens unchanged. Default: 0.1
        ignore_index (int): Index to ignore in loss calculation. Default: -100
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (masked_tokens, labels)
            - masked_tokens: Input tokens with masking applied
            - labels: Original tokens for loss calculation
    """
    
    def __init__(self,
                 x: ArrayLike,
                 bin_ids: ArrayLike,
                 encoding_info: Dict[str, int],
                 mask_token_id: int=0,
                 mask_token_prob: float=0.15,
                 random_token_prob: float=0.1,
                 unchanged_token_prob: float=0.1,
                 ignore_index: int=-100
                 ) -> None:
        
        super(SSLDataset, self).__init__()
        
        # Convert pandas DataFrame to numpy if needed
        if isinstance(x, pd.DataFrame):
            x = x.values
        
        if isinstance(bin_ids, pd.DataFrame):
            bin_ids = bin_ids.values
            
        # Store data and parameters
        self.x = x
        self.bin_ids = bin_ids
        self.encoding_info = encoding_info
        self.mask_token_id = mask_token_id
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.unchanged_token_prob = unchanged_token_prob
        self.ignore_index = ignore_index
        
        # Create the number of bins tensor for each feature
        self.num_bins = torch.tensor([
            v.get('num_bins', v.get('num_categories'))
            for _, v in encoding_info.items()
        ])
        
    def _apply_masking(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking strategy to input tokens.
        
        Args:
            tokens (torch.Tensor): Original token sequence
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (masked_tokens, labels)
        """
        # Clone tokens for labels and masking
        labels = tokens.clone()
        masked_tokens = tokens.clone()
        
        # Generate random probabilities for each token
        probs = torch.rand(tokens.shape)
        
        # Determine which tokens to process (mask_token_prob of all tokens)
        mask_candidates = probs < self.mask_token_prob
        
        # Within mask candidates, determine the specific action:
        # - random_token_prob: replace with random token
        # - unchanged_token_prob: keep original token
        # - remaining: replace with [MASK] token
        
        random_mask = probs < (self.mask_token_prob * self.random_token_prob)
        unchanged_mask = (probs > (self.mask_token_prob - self.mask_token_prob * self.unchanged_token_prob)) & mask_candidates
        mask = mask_candidates & ~(random_mask | unchanged_mask)
        
        # Apply random token replacement
        masked_tokens[random_mask] = (torch.rand(len(tokens)) * self.num_bins + 1).type(masked_tokens.dtype)[random_mask]
        
        # Apply mask token
        masked_tokens[mask] = self.mask_token_id
        
        # Set labels for non-masked tokens to ignore_index
        labels[~mask_candidates] = self.ignore_index
        
        return masked_tokens, labels
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with masking applied.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (masked_tokens, labels)
        """
        # Get tokens for this sample
        y = torch.tensor(self.x[idx], dtype=torch.float)
        tokens = torch.tensor(self.bin_ids[idx], dtype=torch.long)
        
        # Apply masking strategy
        masked_tokens, labels = self._apply_masking(tokens)
        y[labels == self.ignore_index] = torch.nan
        
        return masked_tokens, labels, y
    
    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.bin_ids)



class FinetuneDataset(Dataset):
    def __init__(self, 
                 bin_ids: ArrayLike,
                 y: ArrayLike
                 ) -> None:
        super(FinetuneDataset, self).__init__()
        
        self.bin_ids = bin_ids
        self.y = y
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bin_idx = torch.tensor(self.bin_ids[idx], dtype=torch.long)
        dtype = torch.long if np.issubdtype(self.y[idx].dtype, np.integer) else torch.float
        label = torch.tensor(self.y[idx], dtype=dtype)
        return bin_idx, label
    
    def __len__(self) -> int:
        return len(self.bin_ids)



if __name__ == '__main__':
    x = np.random.rand(20, 10)
    discretizer = QuantileDiscretize(num_bins=100, encoding_info={0: {'num_bins': 10}})
    discretizer.fit(x)
    bin_ids = discretizer.discretize(x)
    dataset = SSLDataset(x=x,
                     bin_ids=bin_ids,
                     encoding_info=discretizer.encoding_info,
                     mask_token_prob=1.0,
                     random_token_prob=0.3,
                     unchanged_token_prob=0.3,
                     ignore_index=-100)
    print(dataset[0])
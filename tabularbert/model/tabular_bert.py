import functools
import json
import os
import warnings
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR 
import pandas as pd
import numpy as np

from .embedding import TokenEmbedding
from .bert import BERT, Classifier, Regressor
from .mlp import MLP
from ..utils.type import ArrayLike
from ..utils.utils import (
    DualLogger, 
    CheckPoint, 
    EarlyStopping, 
    make_save_dir,
    separate_decay_params,
    to_serializable
)
from ..utils.data import (
    QuantileDiscretize, 
    UniformDiscretize, 
    SSLDataset, 
    FinetuneDataset
)
from ..utils.criterion import TabularMSE, TabularWasserstein, TabularCrossEntropy
from ..utils.regularizer import L2Penalty, SquaredL2Penalty
# from ..utils.scheduler import WarmupCosineLR


class TabularBERT(nn.Module):
    """
    TabularBERT: A BERT-based model for tabular data analysis.
    
    This model applies BERT transformer architecture to tabular data by discretizing
    continuous values into bins and treating them as tokens. 
    
    Architecture:
    1. TokenEmbedding: Converts bin indices to embeddings with positional information
    2. BERT: Transformer encoder for learning contextual representations
    3. Classifier: Multi-task classification head for sequence-to-sequence predictions
    4. Regressor: Multi-task regression head for sequence-to-value predictions
    
    Key Features:
    - Handles mixed-type tabular data (continuous and categorical)
    - Multi-task learning with both classification and regression outputs
    - Position-aware embeddings for tabular structure
    - Self-attention mechanism for feature interactions
    
    Args:
        encoding_info (Dict[str, Dict[str, int]]): Nested dictionary containing encoding
                                                   information for each variable/column
        embedding_dim (int): Dimension of embedding vectors. Default: 256
        n_layers (int): Number of transformer encoder layers. Default: 4
        n_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout probability for regularization. Default: 0.1
        mode (str): Mode for concatenating bin and positional embeddings. Default: 'add'
    
    Example:
        >>> encoding_info = {'var1': {'num_bins': 10}, 'var2': {'num_categories': 5}}
        >>> model = TabularBERT(encoding_info)
        >>> bin_ids = torch.randint(1, 11, (32, 2))  # batch_size=32, 2 variables
        >>> reg_outputs, cls_outputs = model(bin_ids)
    """
    
    def __init__(self,
                 encoding_info: Dict[str, Dict[str, int]],
                 embedding_dim: int=256,
                 n_layers: int=3,
                 n_heads: int=8,
                 dropout: float=0.1,
                 mode: str='add') -> None:
        super(TabularBERT, self).__init__()
        
        # Store configuration
        self.encoding_info = encoding_info
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.mode = mode
        
        # Validate inputs
        if embedding_dim % n_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})")
        
        # Initialize token embedding layer
        self.token_embedding = TokenEmbedding(
            encoding_info=encoding_info,
            embedding_dim=embedding_dim,
            dropout=dropout,
            mode=mode
        )
        
        # Initialize BERT encoder
        self.bert = BERT(
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Initialize prediction heads
        self.classifier = Classifier(embedding_dim=embedding_dim,
                                     encoding_info=encoding_info)
        self.regressor = Regressor(embedding_dim=embedding_dim,
                                   encoding_info=encoding_info)
        
    def forward(self, bin_ids: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the TabularBERT model.
        
        Args:
            bin_ids (torch.Tensor): Input tensor of shape (batch_size, num_variables)
                                   containing bin indices for each variable
        
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: A tuple containing:
                - reg_outputs: List of regression outputs (one tensor per variable)
                - cls_outputs: List of classification outputs (one tensor per variable)
        """
        
        contextualized_embeddings = self.get_embeddings(bin_ids)
        
        # Prediction heads: generate task-specific outputs
        cls_outputs = self.classifier(contextualized_embeddings[:, 1:])
        reg_outputs = self.regressor(contextualized_embeddings[:, 1:])
        
        return cls_outputs, reg_outputs
    
    def get_embeddings(self, bin_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract contextualized embeddings without prediction heads.
        
        Useful for feature extraction or transfer learning scenarios.
        
        Args:
            bin_ids (torch.Tensor): Input tensor of shape (batch_size, num_variables)
        
        Returns:
            torch.Tensor: Contextualized embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        
        # Token embedding: bin_ids -> embeddings
        embeddings = self.token_embedding(bin_ids)
        
        # BERT encoder: learn contextual representations
        contextualized_embeddings = self.bert(embeddings)
        return contextualized_embeddings
    
    

class DownstreamModel(nn.Module):
    """
    DownstreamModel: A model for downstream tasks using pretrained contextualized embeddings.
    
    Args:
        pretrained (TabularBERT): Pretrained TabularBERT model
        head (MLP): MLP head for downstream tasks
    """
    def __init__(self, pretrained: TabularBERT, head: MLP):
        super(DownstreamModel, self).__init__()
        self.encoding_info = pretrained.encoding_info
        self.pretrained = pretrained
        self.head = head
    
    def forward(self, bin_ids: torch.Tensor) -> torch.Tensor:
        contextualized_embeddings = self.pretrained.get_embeddings(bin_ids)
        outputs = self.head(contextualized_embeddings[:, 0])
        return outputs



class TabularBERTTrainer(nn.Module):
    """
    TabularBERTTrainer: A comprehensive training framework for TabularBERT models.
    
    This class manages the complete training workflow for tabular data analysis,
    including both self-supervised pretraining and supervised fine-tuning phases.
    It handles data preprocessing, model initialization, and training orchestration.
    
    Training Phases:
    1. Self-Supervised Pretraining: Uses masked language modeling on tabular data
       to learn meaningful representations without labels
    2. Supervised Fine-tuning: Adapts the pretrained model for specific downstream
       tasks (classification/regression) using labeled data
    
    Key Features:
    - Automated data preprocessing and binning
    - Self-supervised pretraining with masking strategies
    - Multi-task fine-tuning for classification and regression
    - Flexible encoding schemes for different data types
    
    Args:
        x (ArrayLike): Training data for pretraining and fine-tuning array/matrix
        y (ArrayLike): Training labels for fine-tuning array/matrix 
        num_bins (int): Number of bins for discretization
        encoding_info Dict[str, int]: 
                      Encoding configuration for variables. Can be:
                      - Dict mapping variable names to number of bins
                      - Omitted variables are discretized into the default number of bins (num_bins)
                      - None: All variables are discretized into the default number of bins (num_bins)
        device (torch.device): Device for computation. Default: CPU
        valid_x (ArrayLike, optional): Validation data for monitoring pretraining and fine-tuning
        valid_y (ArrayLike, optional): Validation labels for monitoring fine-tuning
    
    Example:
        >>> import torch
        >>> import numpy as np
        >>> 
        >>> # Prepare data
        >>> x_train = np.random.randn(1000, 5)  # 1000 samples, 5 features
        >>> encoding_info = {'var1': {'num_bins': 20}, 'var5': 'categorical'}
        >>> 
        >>> # Initialize trainer
        >>> trainer = TabularBERTTrainer(
        ...     x=x_train,
        ...     y=y_train,
        ...     num_bins=50,
        ...     encoding_info=encoding_info,
        ...     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ... )
        >>> 
        >>> # Run pretraining
        >>> trainer.pretrain(epochs=100)
        >>> 
        >>> # Fine-tune for downstream task
        >>> trainer.finetune(epochs=50)
    """
    
    def __init__(
        self,
        x: ArrayLike=None,
        y: ArrayLike=None,
        num_bins: int=50,
        encoding_info: Dict[str, str|int]=None,
        valid_x: ArrayLike=None,
        valid_y: ArrayLike=None,
        device: torch.device=torch.device('cpu')
    ) -> None:
        super(TabularBERTTrainer, self).__init__()
        
        # Store input data and configuration
        self.x = x
        self.y = y
        self.num_bins = num_bins
        self.encoding_info = encoding_info
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.device = device
        self.bin_ids = None
        self.valid_bin_ids = None
        
        # Handle encoding_info: ensure it's not empty dict
        encoding_info = encoding_info if encoding_info else None
        
        # Initialize discretizer and process data
        if x is not None:
            self.discretizer = UniformDiscretize(
                num_bins=num_bins, 
                encoding_info=encoding_info
            )
            self.discretizer.fit(x)
        
            # Discretize training data
            self.bin_ids = self.discretizer.discretize(x)
            
            # Discretize validation data if provided
            self.valid_bin_ids = (
                self.discretizer.discretize(valid_x) 
                if valid_x is not None else None
            )
        
        # Initialize model components (will be set later)
        self.model = None
        self.head = None
        self.optimizer = None
        self.save = False
    
    def setup_directories_and_logging(self, 
                                      save_dir: str,
                                      phase: str="pretraining",
                                      project_name: str="tabular-bert",
                                      experiment_name: str=None,
                                      use_wandb: bool=True) -> None:
        """
        Setup save directories and logging infrastructure.
        
        Creates necessary directories for model checkpoints and initializes
        dual logging (TensorBoard + WandB) for comprehensive experiment tracking.
        
        Args:
            save_dir (str): Directory to save model configurations and checkpoints
            project_name (str): WandB project name. Default: "tabular-bert"
            experiment_name (str, optional): Experiment name for WandB run
            use_wandb (bool): Whether to use WandB logging. Default: True
        """
        
        assert phase in ['pretraining', 'fine-tuning'], "Phase must be 'pretraining' or 'fine-tuning'"
        
        # Create save directory for pretraining/finetuning
        self.save_dir = make_save_dir(save_dir)
        
        # Initialize configuration dictionary for comprehensive tracking
        self.config = {
            'project': project_name,
            # Data configuration
            'data': {
                'num_bins': self.num_bins,
                'encoding_info': self.encoding_info,
                'data_shape': self.x.shape if hasattr(self.x, 'shape') else None,
                'has_validation': self.valid_x is not None,
                'validation_shape': self.valid_x.shape if self.valid_x is not None else None
            },
            # System configuration
            'system': {
                'device': str(self.device)
            },
            # Pretraining/Finetuning configuration
            phase: {'model': {},
                    'optimizer': {},
                    'training': {}
                    }
        }
        
        # Save initial configuration as JSON
        self._save_config()
        
        # Initialize dual logger (TensorBoard + WandB)
        log_dir = os.path.join(self.save_dir, 'log')
        self.logger = DualLogger(
            log_dir=log_dir,
            project_name=project_name,
            experiment_name=experiment_name,
            config=self.config,
            use_wandb=use_wandb
        )
        self.save = True
        self.phase = phase
    
    def _save_config(self) -> None:
        """
        Save the current configuration to a JSON file.
        
        Creates a comprehensive configuration file that includes all model,
        optimizer, and training parameters for reproducibility.
        """
        config_path = os.path.join(self.save_dir, 'config.json')
        
        # Create a serializable copy of the config
        serializable_config = to_serializable(self.config.copy())
        
        try:
            with open(config_path, 'w') as f:
                json.dump(serializable_config, f, indent=4, sort_keys=True)
        except Exception as e:
            warnings.warn(f"Could not save config to {config_path}: {e}")
    
    def set_bert(self, 
                embedding_dim: int=1024,
                n_layers: int=3,
                n_heads: int=8,
                dropout: float=0.1,
                mode: str='add') -> None:
        """
        Initialize the TabularBERT model for masked language modeling.
        
        Args:
            embedding_dim (int): Dimension of embedding vectors. Default: 1024
            n_layers (int): Number of transformer encoder layers. Default: 3
            n_heads (int): Number of attention heads. Default: 8
            dropout (float): Dropout probability for regularization. Default: 0.1
        """
        
        # Update model configuration
        if self.save:
            self.config['pretraining']['model'] = {
                'architecture': 'TabularBERT',
                'embedding_dim': embedding_dim,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'dropout': dropout,
                'mode': mode,
                'total_parameters': None  # Will be updated after model creation
            }
        
        # Create model
        self.model = TabularBERT(
            encoding_info=self.discretizer.encoding_info,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            mode=mode
        ).to(self.device)
        
        # Update total parameters count
        if self.save:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.config['pretraining']['model']['total_parameters'] = total_params
            self.config['pretraining']['model']['trainable_parameters'] = trainable_params
        
            # Save updated configuration
            self._save_config()
    
    def set_head(self,
                 output_dim: int=None,
                 hidden_layers: List[int]=None,
                 activation: str='ReLU',
                 dropouts: float | List[float]=0.1,
                 batch_norm: bool=False
                 ) -> None:
        """
        Configure the head for fine-tuning.
        
        Args:
            output_dim (int): Output dimension for the head. Default: Automatically inferred from the task type.
            hidden_layers (List[int]): Hidden dimensions for the head. Default: [Embedding Dimension]
            activation (nn.Module): Activation function for the head. Default: ReLU
            dropouts (float | List[float]): Dropout probabilities for the head. Default: 0.1
            batch_norm (bool): Whether to use batch normalization. Default: False
        """
        
        if output_dim is None:
            output_dim = self.output_dim
        else:
            self.output_dim = output_dim
        
        # Update model configuration
        hidden_layers = [] if hidden_layers is None else hidden_layers
        if self.save:
            self.config['fine-tuning']['model']['head'] = {
                'output_dim': output_dim,
                'hidden_layers': hidden_layers,
                'activation': activation,
                'dropouts': dropouts,
                'batch_norm': batch_norm
            }
        
        self.head = MLP(input_dim=self.model.embedding_dim,
                       output_dim=output_dim,
                       hidden_layers=hidden_layers,
                       activation=activation,
                       dropouts=dropouts,
                       batch_norm=batch_norm
                       ).to(self.device)
        
        # Save updated configuration
        if self.save:
            self._save_config()
        
    def set_optimizer(self,
                      lr: float=2e-4,
                      weight_decay: float=1e-3,
                      betas: Tuple[float, float]=(0.9, 0.999)
                      ) -> None:
        """
        Configure the optimizer for training.
        
        Args:
            lr (float): Learning rate. Default: 1e-4
            weight_decay (float): Weight decay for regularization. Default: 0.001
            betas (Tuple[float, float]): Adam beta parameters. Default: (0.9, 0.999)
        """
        # Update optimizer configuration
        if self.save:
            self.config[self.phase]['optimizer'] = {
                'type': 'AdamW',
                'lr': lr,
                'weight_decay': weight_decay,
                'betas': betas
            }
        
        # Create optimizer partial function
        self.optimizer = functools.partial(
            torch.optim.AdamW, 
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
        
        # Save updated configuration
        if self.save:
            self._save_config()
    
    def pretrain(self, 
              epochs: int=1000,
              batch_size: int=256,
              penalty: str='squaredL2',
              lamb: float=0.5,
              mask_token_id: int=0,
              mask_token_prob: float=0.2,
              random_token_prob: float=0.15,
              unchanged_token_prob: float=0.15,
              ignore_index: int=-100,
              num_workers: int=0,
              ) -> None:
        """
        Run self-supervised pretraining using masked language modeling.
        
        Args:
            epochs (int): Number of training epochs. Default: 1000
            batch_size (int): Batch size for training. Default: 256
            penalty (str): Penalty type for embedding regularization ('L2' or 'squaredL2'). Default: 'squaredL2'
            lamb (float): Regularization parameter. Default: 0.5
            mask_token_id (int): Token ID used for masking. Default: 0
            mask_token_prob (float): Probability of replacing tokens with [MASK]. Default: 0.2
            random_token_prob (float): Probability of replacing tokens with random values. Default: 0.15
            unchanged_token_prob (float): Probability of keeping original tokens unchanged. Default: 0.15
            ignore_index (int): Index to ignore in loss calculation. Default: -100
            num_workers (int): Number of subprocesses to use for data loading. Default: 0
        """
        # Update training configuration
        if self.save:
            self.config['pretraining']['training'] = {
                'epochs': epochs,
                'batch_size': batch_size,
                'penalty': penalty,
                'regularization_lambda': lamb,
                'masking': {
                    'mask_token_id': mask_token_id,
                    'mask_token_prob': mask_token_prob,
                    'random_token_prob': random_token_prob,
                    'unchanged_token_prob': unchanged_token_prob
                },
                'ignore_index': ignore_index,
                'num_workers': num_workers
            }
            # Save updated configuration before training starts
            self._save_config()
        
        train_dataset = SSLDataset(
            x = self.x,
            bin_ids=self.bin_ids,
            encoding_info=self.discretizer.encoding_info,
            mask_token_id=mask_token_id,
            mask_token_prob=mask_token_prob,
            random_token_prob=random_token_prob,
            unchanged_token_prob=unchanged_token_prob,
            ignore_index=ignore_index
        )
        trainloader = DataLoader(train_dataset, 
                                 batch_size=batch_size,
                                 shuffle=True, 
                                 drop_last=True,
                                 num_workers=num_workers)

        if self.valid_x is not None:
            # transformed_valid_x = scaler.transform(self.valid_x)
            valid_dataset = SSLDataset(
                x = self.valid_x,
                bin_ids=self.valid_bin_ids,
                encoding_info=self.discretizer.encoding_info,
                mask_token_id=mask_token_id,
                mask_token_prob=mask_token_prob,
                random_token_prob=random_token_prob,
                unchanged_token_prob=unchanged_token_prob,
                ignore_index=ignore_index
            )

            validloader = DataLoader(valid_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True,
                                     drop_last=False,
                                     num_workers=num_workers)
            
        if self.model is None:
            warnings.warn(
                """
                TabularBERT Model Auto-Configuration\n
                ======================================================================\n
                No model configuration detected. Initializing with optimized defaults:\n\n
                Architecture: TabularBERT Transformer\n
                Embedding Dimension: 1024\n
                Transformer Layers: 3\n
                Attention Heads: 8\n
                Dropout Rate: 0.1\n\n
                Tip: Use trainer.set_bert() to customize architecture before training.\n
                ======================================================================
                """,
                UserWarning
            )
            self.set_bert()
        
        # Define loss functions
        mse_loss = TabularMSE(self.discretizer.encoding_info)
        wasserstein_loss = TabularWasserstein(self.discretizer.encoding_info, ignore_index=ignore_index)
        ce_loss = TabularCrossEntropy(self.discretizer.encoding_info, ignore_index=ignore_index)
        
        # Define regularizer
        if penalty == 'L2':
            embed_penalty = L2Penalty(lamb)
        elif penalty == 'squaredL2':
            embed_penalty = SquaredL2Penalty(lamb)
        self.lamb = lamb
        self.penalty = penalty
        
        # Define checkpoint
        if self.save:
            checkpoint = CheckPoint(self.save_dir, phase='pretraining', max=False)
        
        # Define optimizer
        if self.optimizer is None:
            warnings.warn(
                """
                TabularBERT Optimizer Auto-Configuration\n
                ======================================================================\n
                No optimizer configuration detected. Initializing with optimized defaults:\n\n
                Optimizer: AdamW\n
                Learning Rate: 2e-4\n
                Weight Decay: 1e-2\n
                Beta Parameters: (0.9, 0.999)\n\n
                Tip: Use trainer.set_optimizer() to customize optimizer before training.\n
                ======================================================================
                """,
                UserWarning
            )
            self.set_optimizer(weight_decay=1e-2)
        
        decay_params, no_decay_params = separate_decay_params(self.model)    
        optimizer = self.optimizer([
            {'params': decay_params},
            {'params': no_decay_params, 'weight_decay': 0.0}
            ])
        total_steps = epochs * len(trainloader)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop with progress tracking
        global_step = 0
        
        print(f"\n Starting TabularBERT Pretraining")
        print(f"{'='*60}")
        print(f"Dataset: {len(trainloader)} batches ({len(trainloader.dataset)} samples)")
        print(f"Epochs: {epochs} | Batch Size: {batch_size}")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"{'='*60}\n")
        
        for epoch in tqdm.tqdm(range(epochs), desc='Pretraining Progress'):
            # Training phase
            train_metrics = self._run_pretraining_epoch(
                optimizer, scheduler, trainloader, mse_loss, wasserstein_loss, ce_loss, embed_penalty, global_step
            )
            global_step = train_metrics['global_step']
            
            # Validation phase (if validation data available)
            if self.valid_x is not None:
                valid_metrics = self._run_pretraining_validation_epoch(
                    validloader, mse_loss, wasserstein_loss, ce_loss, embed_penalty, epoch
                )
                
                # Model checkpointing based on validation loss
                current_loss = valid_metrics['risk']
                if self.save:
                    checkpoint(current_loss, self.discretizer, self.model, self.config)
                
                # Progress reporting
                self._log_epoch_progress(train_metrics['risk'], valid_metrics['risk'])
            else:
                # No validation data - checkpoint on training loss
                # current_loss = train_metrics['risk']
                if self.save:
                    # checkpoint(current_loss, self.model, self.config)
                    checkpoint._save_checkpoint(self.discretizer, self.model, self.config)
                
                # Training-only progress reporting
                self._log_epoch_progress(train_metrics['risk'])
        
        print(f"\n Pretraining completed!")
        if self.save:
            print(f"\n Model saved to: {self.save_dir}")
        
        # Reset
        self.save = False
        self.optimizer = None
    
    def _run_pretraining_epoch(self, optimizer, scheduler, trainloader, mse_loss, wasserstein_loss, ce_loss,
                               embed_penalty, epoch):
        """
        Execute one pretraining epoch with efficient batch processing.
        
        Returns:
            dict: Training metrics including total loss and global step
        """
        self.model.train()
        total_loss = 0.0
        pred_loss = 0.0
        num_batches = len(trainloader)
        
        for batch_idx, (batch_bin_ids, batch_labels, batch_tabular_x) in enumerate(trainloader):
            batch_bin_ids = batch_bin_ids.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_tabular_x = batch_tabular_x.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            cls_predictions, reg_predictions = self.model(batch_bin_ids)
            
            # Compute losses
            mse_loss_val = mse_loss(reg_predictions, batch_tabular_x)
            wasserstein_loss_val = wasserstein_loss(cls_predictions, batch_labels)
            regularization_loss = embed_penalty(self.model.token_embedding.num_embedding.embedding.weight) if self.model.token_embedding.num_embedding is not None else 0.0
            
            # Combined loss
            pred_batch_loss = mse_loss_val + wasserstein_loss_val
            total_batch_loss = pred_batch_loss + regularization_loss
            
            # Backward pass and optimization
            total_batch_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Metrics tracking
            total_loss += total_batch_loss.item()
            pred_loss += pred_batch_loss.item()
            
            # Log detailed metrics
            if self.save:
                self.logger.log_scalar('Loss/Train/Total', total_batch_loss.item(), epoch)
                self.logger.log_scalar('Loss/Train/Wasserstein', wasserstein_loss_val.item(), epoch)
                self.logger.log_scalar('Loss/Train/MSE', mse_loss_val.item(), epoch)
                self.logger.log_scalar('Loss/Train/Regularization', regularization_loss.item(), epoch)
            
            epoch += 1
        
        return {
            'risk': pred_loss / num_batches,
            'global_step': epoch,
        }
    
    def _run_pretraining_validation_epoch(self, validloader, mse_loss, wasserstein_loss, ce_loss,
                                          embed_penalty, epoch):
        """
        Execute one pretraining validation epoch with no gradient computation.
        
        Returns:
            dict: Validation metrics including average loss
        """
        self.model.eval()
        
        with torch.no_grad():
            cls_predictions = []
            reg_predictions = []
            labels = []
            tabular_x = []
            for batch_idx, (batch_bin_ids, batch_labels, batch_tabular_x) in enumerate(validloader):
                batch_bin_ids = batch_bin_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_tabular_x = batch_tabular_x.to(self.device)
                
                # Forward pass only
                predictions = self.model(batch_bin_ids)
                cls_predictions.append(predictions[0])
                reg_predictions.append(predictions[1])
                labels.append(batch_labels)
                tabular_x.append(batch_tabular_x)
            
            cls_predictions = [torch.cat(cls_pred, dim=0) for cls_pred in zip(*cls_predictions)]
            reg_predictions = [torch.cat(reg_pred, dim=0) if any(t is not None for t in reg_pred) else None
                               for reg_pred in zip(*reg_predictions)]
            labels = torch.cat(labels, dim=0)
            tabular_x = torch.cat(tabular_x, dim=0)
                
            # Compute losses
            mse_loss_val = mse_loss(reg_predictions, tabular_x)
            wasserstein_loss_val = wasserstein_loss(cls_predictions, labels)
            regularization_loss = embed_penalty(self.model.token_embedding.num_embedding.embedding.weight) if self.model.token_embedding.num_embedding is not None else 0.0
            
            # Combined loss
            pred_risk = mse_loss_val + wasserstein_loss_val
            total_risk = pred_risk + regularization_loss 
            
            # Log validation metrics
            if self.save:
                self.logger.log_scalar('Loss/Valid/Total', total_risk, epoch)
                self.logger.log_scalar('Loss/Valid/MSE', mse_loss_val.item(), epoch)
                self.logger.log_scalar('Loss/Valid/Wasserstein', wasserstein_loss_val.item(), epoch)    
                
        return {
            'risk': pred_risk,
        }
    
    def _log_epoch_progress(self, train_loss, valid_loss=None, train_metric=None, valid_metric=None):
        """
        Log epoch loss information (progress bar handled by tqdm).
        """
        if valid_loss is not None:
            loss_info = f"Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}"
        else:
            loss_info = f"Train Loss: {train_loss:.6f}"
        
        if train_metric is not None:
            loss_info += f" | Train Metric: {train_metric:.6f}"
        
        if valid_metric is not None:
            loss_info += f" | Valid Metric: {valid_metric:.6f}"
        
        print(f"  {loss_info}")    
    
    @classmethod                
    def from_pretrained(cls, save_path, device):
        """
        Load a pre-trained TabularBERT model from a checkpoint.
        
        This method properly loads a pretrained model with device mapping,
        configuration validation, and state restoration for fine-tuning.
        
        Args:
            save_path (str): Path to the pre-trained model checkpoint file
        
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
            ValueError: If the model configuration is incompatible
        """
        
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint file not found: {save_path}")
        
        # Load the pretrained model configuration
        config = torch.load(save_path)
        num_bins = config['data_config']['num_bins']
        encoding_info = config['data_config']['encoding_info']
        lamb = config['regularization_lambda']
        penalty = config['penalty']
        
        trainer = cls(num_bins = num_bins,
                      encoding_info = encoding_info,
                      device = device)
        
        trainer.lamb = lamb
        trainer.penalty = penalty
        
        # Load discretizer
        trainer.discretizer = UniformDiscretize(num_bins, encoding_info)
        trainer.discretizer.bins = config['discretizer']['bins']
        trainer.discretizer.category_maps = config['discretizer']['category_maps']
        
        # Load pretrained model
        trainer.model = TabularBERT(**config['model_config'])
        trainer.model.load_state_dict(config['model_state_dict'])
        trainer.model.to(device)
        
        print(f"Successfully loaded pretrained model from: {save_path}")
        
        return trainer
        
    def finetune(self, 
                 x: ArrayLike=None,
                 y: ArrayLike=None,
                 valid_x: ArrayLike=None,
                 valid_y: ArrayLike=None,
                 num_bins: int=None,
                 encoding_info: Dict[str, Dict[str, int]]=None,
                 task_type: str=None,
                 num_classes: int=None,
                 epochs: int=1000,
                 batch_size: int=256,
                 penalty: str=None,
                 lamb: float=None,
                 criterion: nn.Module=None,
                 metric: nn.Module=None,
                 patience: int=20,
                 num_workers: int=0
                 ) -> None: 
        """
        Fine-tune the TabularBERT model on a downstream task.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            lamb (float): Lambda parameter for the loss function
            num_workers (int): Number of workers for data loading
        """
        
        if num_bins is None:
            num_bins = self.num_bins
        if encoding_info is None:
            encoding_info = self.encoding_info
        
        if x is None:
            if self.bin_ids is None:
                raise ValueError("x is not provided.")
            bin_ids = self.bin_ids
        else:
            discretizer = UniformDiscretize(
                num_bins=num_bins, 
                encoding_info=encoding_info
            )
            discretizer.fit(x)
            # Discretize training data
            bin_ids = discretizer.discretize(x)
        
        if valid_x is None:
            if self.valid_bin_ids is not None:
                valid_bin_ids = self.valid_bin_ids    
            else:
                valid_bin_ids = None
        else:
            valid_bin_ids = discretizer.discretize(valid_x)
        
        if self.save:
            self.config['data'] = {
                'num_bins': num_bins,
                'encoding_info': encoding_info,
                'data_shape': bin_ids.shape if hasattr(bin_ids, 'shape') else None,
                'has_validation': valid_bin_ids is not None,
                'validation_shape': valid_bin_ids.shape if valid_bin_ids is not None else None
            }
            self._save_config()
        
        # Convert pandas objects to numpy arrays
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            target_y = y.values 
        else:
            target_y = y
        
        # Process labels
        target_y, label_stats = self._process_labels(target_y)
        if task_type is None:
            warnings.warn(
                "Task type not specified. Automatically inferring task type from the labels."
            )
            task_type = label_stats['task_type']
        
        if num_classes is None:
            warnings.warn(
                "Number of classes not specified. Automatically inferring number of classes from the labels."
            )
            num_classes = label_stats['num_classes']
        
        if criterion is None:
            warnings.warn(
                "Criterion not specified. Automatically inferring criterion from the task type."
            )
            if task_type == 'classification':
                criterion = nn.CrossEntropyLoss(ignore_index=self.config['pretraining']['training']['ignore_index'])
            elif task_type == 'regression':
                criterion = nn.MSELoss()
        
        if lamb is None:
            lamb = self.lamb
        if penalty is None:
            penalty = self.penalty
        
        # Update training configuration
        if self.save:
            self.config['fine-tuning']['training'] = {
                'task_type': task_type,
                'num_classes': num_classes,
                'epochs': epochs,
                'batch_size': batch_size,
                'penalty': penalty,
                'regularization_lambda': lamb,
                'criterion': criterion,
                'num_workers': num_workers
            }
            # Save updated configuration before training starts
            self._save_config()
        
        train_dataset = FinetuneDataset(bin_ids, target_y)
        trainloader = DataLoader(train_dataset, 
                                 batch_size=batch_size,
                                 shuffle=True, 
                                 drop_last=True,
                                 num_workers=num_workers)
        
        if valid_bin_ids is not None:
            if valid_y is None:
                raise ValueError("No labels found in the validation dataset")

            if isinstance(valid_y, pd.Series) or isinstance(valid_y, pd.DataFrame):
                valid_target_y = valid_y.values 
            else:
                valid_target_y = valid_y
                
            valid_target_y, _ = self._process_labels(valid_target_y, label_stats['reference'])
            valid_dataset = FinetuneDataset(valid_bin_ids, valid_target_y)
            validloader = DataLoader(valid_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True,
                                     drop_last=False,
                                     num_workers=num_workers)
            
        if self.model is None:
            raise ValueError(
                "Pretrained model required for fine-tuning. "
                "Please pretrain the model by running pretraining with the 'pretrain()' method "
                "or load a pretrained model using the 'from_pretrained()' method."
            )
        
        if self.head is None:
            self.output_dim = num_classes if task_type == 'classification' else target_y.shape[1]
            warnings.warn(
                """
                TabularBERT Model Auto-Configuration\n
                ======================================================================\n
                No head configuration detected. Initializing with optimized defaults:\n\n
                Architecture: MLP\n
                Input Dimension: Embedding Dimension\n
                Output Dimension: {output_dim}\n
                Hidden Layers: 1\n
                Hidden Layer Dimensions: Embedding Dimension\n
                Activation Function: ReLU\n
                Dropout Rate: 0.1\n
                Batch Normalization: False\n\n
                Tip: Use trainer.set_head() to customize head before training.\n
                ======================================================================
                """.format(output_dim = self.output_dim),
                UserWarning
            )
            self.set_head()
        
        # Define regularizer
        if penalty == 'L2':
            embed_penalty = L2Penalty(lamb)
        elif penalty == 'squaredL2':
            embed_penalty = SquaredL2Penalty(lamb)
        
        # Define checkpoint
        if self.save:
            checkpoint = CheckPoint(self.save_dir, phase='finetuning', max=False)
        
        # Define early stopping
        if patience is not None:
            early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # Define model
        self.model = DownstreamModel(pretrained=self.model,
                                     head = self.head)
        
        # Define optimizer
        if self.optimizer is None:
            warnings.warn(
                """
                TabularBERT Optimizer Auto-Configuration\n
                ======================================================================\n
                No optimizer configuration detected. Initializing with optimized defaults:\n\n
                Optimizer: AdamW\n
                Learning Rate: 2e-4\n
                Weight Decay: 1e-5\n
                Beta Parameters: (0.9, 0.999)\n\n
                Tip: Use trainer.set_optimizer() to customize optimizer before training.\n
                ======================================================================
                """,
                UserWarning
            )
            self.set_optimizer(weight_decay=1e-5)
        
        decay_params, no_decay_params = separate_decay_params(self.model)    
        optimizer = self.optimizer([
                {'params': decay_params},
                {'params': no_decay_params, 'weight_decay': 0.0}
                ])
        total_steps = epochs * len(trainloader)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop with progress tracking
        global_step = 0
        
        print(f"\n Starting TabularBERT Fine-tuning")
        print(f"{'='*60}")
        print(f"Dataset: {len(trainloader)} batches ({len(trainloader.dataset)} samples)")
        print(f"Epochs: {epochs} | Batch Size: {batch_size}")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"{'='*60}\n")
        
        for epoch in tqdm.tqdm(range(epochs), desc='Fine-tuning Progress'):
            # Training phase
            train_metrics = self._run_finetuning_epoch(
                optimizer, scheduler, trainloader, criterion, metric, embed_penalty, global_step
            )
            global_step = train_metrics['global_step']
            
            # Validation phase (if validation data available)
            if valid_bin_ids is not None:
                valid_metrics = self._run_finetuning_validation_epoch(
                    validloader, criterion, metric, embed_penalty, epoch
                )
                
                # Model checkpointing based on validation loss
                if self.save:
                    if metric is not None:
                        checkpoint(valid_metrics['metric'], self.discretizer, self.model, self.config)
                    else:
                        checkpoint(valid_metrics['risk'], self.discretizer, self.model, self.config)
                    
                # Progress reporting
                self._log_epoch_progress(train_metrics['risk'], valid_metrics['risk'],
                                         train_metrics['metric'], valid_metrics['metric'])
                
                if patience is not None:
                    if metric is not None:
                        stop = early_stopping(valid_metrics['metric'])
                    else:
                        stop = early_stopping(valid_metrics['risk'])
                    if stop:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                # No validation data - checkpoint on training loss
                if self.save:
                    # if metric is not None:
                    #     checkpoint(train_metrics['metric'], self.model, self.config)
                    # else:
                    #     checkpoint(train_metrics['risk'], self.model, self.config)
                    checkpoint._save_checkpoint(self.discretizer, self.model, self.config)
                
                # Training-only progress reporting
                self._log_epoch_progress(train_metrics['risk'],
                                         train_metric = train_metrics['metric'])
        
        print(f"\n Fine-tuning completed!")
        if self.save:
            print(f"Model saved to: {self.save_dir}")
        self.save = False
        
    def _run_finetuning_epoch(self, optimizer, scheduler, trainloader, criterion, metric, embed_penalty, epoch):
        """
        Execute one fine-tuning epoch with efficient batch processing.
        
        Returns:
            dict: Training metrics including total loss and global step
        """
        self.model.train()
        total_loss = 0.0
        pred_loss = 0.0
        avg_metric = 0.0
        num_batches = len(trainloader)
        
        for batch_idx, (batch_bin_ids, batch_labels) in enumerate(trainloader):
            batch_bin_ids = batch_bin_ids.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(batch_bin_ids)
            
            # Compute losses
            loss_val = criterion(predictions, batch_labels)
            regularization_loss = embed_penalty(self.model.pretrained.token_embedding.num_embedding.embedding.weight) if self.model.pretrained.token_embedding.num_embedding is not None else 0.0
            
            # Combined loss
            total_batch_loss = loss_val + regularization_loss
            
            # Backward pass and optimization
            total_batch_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Metrics tracking
            total_loss += total_batch_loss.item()
            pred_loss += loss_val.item()
            if metric is not None:
                m = metric(predictions, batch_labels)
                avg_metric += m.item() / num_batches
            
            # Log detailed metrics
            if self.save:
                self.logger.log_scalar('Loss/Train/Total', total_batch_loss.item(), epoch)
                self.logger.log_scalar('Loss/Train/Loss', loss_val.item(), epoch)
                self.logger.log_scalar('Loss/Train/Regularization', regularization_loss.item(), epoch)
                if metric is not None:
                    self.logger.log_scalar('Metric/Train', avg_metric, epoch)
            
            epoch += 1
        
        return {
            'risk': pred_loss / num_batches,
            'metric': avg_metric if metric is not None else None,
            'global_step': epoch
        }    
        
    def _run_finetuning_validation_epoch(self, validloader, criterion, metric, embed_penalty, epoch):
        """
        Execute one fine-tuning validation epoch with no gradient computation.
        
        Returns:
            dict: Validation metrics including average loss
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = []
            labels = []
            for batch_idx, (batch_bin_ids, batch_labels) in enumerate(validloader):
                batch_bin_ids = batch_bin_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass only
                predictions.append(self.model(batch_bin_ids))
                labels.append(batch_labels)
            
            predictions = torch.cat(predictions, dim=0)
            labels = torch.cat(labels, dim=0)
            
            # Compute losses
            loss_val = criterion(predictions, labels)
            regularization_loss = embed_penalty(self.model.pretrained.token_embedding.num_embedding.embedding.weight) if self.model.pretrained.token_embedding.num_embedding is not None else 0.0
            
            # Combined loss
            risk = loss_val + regularization_loss
            
            # Metrics tracking
            if metric is not None:
                m = metric(predictions, labels)
                
            # Log validation metrics
            if self.save:
                self.logger.log_scalar('Loss/Valid/Total', risk.item(), epoch)
                self.logger.log_scalar('Loss/Valid/Loss', loss_val.item(), epoch)
                if metric is not None:
                    self.logger.log_scalar('Metric/Valid', m.item(), epoch)
        
        return {
            'risk': loss_val.item(),
            'metric': m.item() if metric is not None else None,
        }
        
    def _process_labels(self, y: ArrayLike, reference: List | ArrayLike=None) -> Tuple[ArrayLike, Dict[str, int]]:
        """
        Process and analyze labels to determine task type and prepare data.
        
        This method handles:
        - Automatic task type detection (classification vs regression)
        - String/object label encoding for classification
        - Label statistics for output dimension determination
        
        Args:
            y (ArrayLike): Raw labels (can be numpy array)
            reference (List | ArrayLike): Reference labels for classification
        """
        
        if reference is not None:
            map = {k: i for i, k in enumerate(reference)}
            mapping = np.vectorize(map.get)
            y = mapping(y)
            task = 'classification'
            num_classes = len(reference)
            
        else:
            # Determine task type based on data type
            if np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.object_) or np.issubdtype(y.dtype, np.str_):
                reference, y = np.unique(y, return_inverse=True)
                reference = list(reference)
                task = 'classification'
                num_classes = len(reference)
                
            elif np.issubdtype(y.dtype, np.floating):
                task = 'regression'
                num_classes = None
                reference = None
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
            
            else:
                raise ValueError(f"Unsupported label dtype: {y.dtype}. "
                            "Labels must be integer, float, string, or object type.")
        
        # Store label statistics
        label_stats = {
            'task_type': task,
            'num_classes': num_classes,
            'reference': reference
        }
        
        return y, label_stats
        
    @classmethod                
    def from_finetuned(cls, save_path, device):
        """
        Load a fine-tuned TabularBERT downstream model from a checkpoint.
        
        This method properly loads a pretrained model with device mapping,
        configuration validation, and state restoration for fine-tuning.
        
        Args:
            save_path (str): Path to the pre-trained model checkpoint file
        
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
            ValueError: If the model configuration is incompatible
        """
        
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint file not found: {save_path}")
        
        config = torch.load(save_path)
        
        # Load discretizer
        discretizer = UniformDiscretize(config['data']['num_bins'], 
                                        config['data']['encoding_info'])
        discretizer.bins = config['discretizer']['bins']
        discretizer.category_maps = config['discretizer']['category_maps']
        
        # Load the pretrained model configuration
        pretrained = TabularBERT(**config['model_config']['tabular_bert'])
        head = MLP(**config['model_config']['mlp_head'])
        model = DownstreamModel(pretrained, head)
        model.load_state_dict(config['model_state_dict'])
        # model.to(device)
        predictor = TabularBERTPredictor(model, discretizer, device)
        print(f"Successfully loaded fine-tuned model from: {save_path}")
        
        return predictor
        
        

class TabularBERTPredictor:
    """
    TabularBERTPredictor: Inference wrapper for fine-tuned TabularBERT models.
    
    This class combines a fine-tuned downstream model with the fitted discretizer
    used during training. It accepts raw tabular input data, discretizes it into
    bin indices, and runs mini-batch inference through the downstream model.
    
    Args:
        model (DownstreamModel): Fine-tuned downstream model.
        discretizer (DiscretizeBase): Fitted discretizer used to transform raw
                                      tabular values into bin indices.
        device (torch.device): Device used for model inference.
    
    Example:
        >>> predictor = TabularBERTPredictor.from_finetuned(
        ...     save_path="./fine-tuning/version0/model_checkpoint.pt",
        ...     device=torch.device("cuda:0")
        ... )
        >>> predictions = predictor.predict(test_x, batch_size=4096)
    """
    def __init__(self, model, discretizer, device):
        self.model = model
        self.discretizer = discretizer
        self.device = device

    def predict(self, x, batch_size=4096, return_numpy=True):
        """
        Predict outputs for raw tabular input data.
        
        Args:
            x (ArrayLike): Raw tabular input data with shape
                           (num_samples, num_features).
            batch_size (int): Number of samples processed per inference batch.
                              Default: 4096.
            return_numpy (bool): Whether to return predictions as a NumPy array.
                                 If False, returns a CPU torch.Tensor.
                                 Default: True.
        
        Returns:
            np.ndarray | torch.Tensor: Model predictions for all input samples.
        """
        self.model.eval()
        outputs = []

        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                batch_x = x[start:start + batch_size]

                batch_bin_ids = self.discretizer.discretize(batch_x)
                batch_bin_ids = torch.as_tensor(
                    batch_bin_ids,
                    dtype=torch.long,
                    device=self.device
                )
                batch_pred = self.model(batch_bin_ids)
                outputs.append(batch_pred.detach().cpu())

        outputs = torch.cat(outputs, dim=0)

        if return_numpy:
            return outputs.numpy()
        return outputs
    
    @classmethod                
    def from_finetuned(cls, save_path, device):
        """
        Load a fine-tuned TabularBERT predictor from a checkpoint.
        
        This method restores both the fitted discretizer and the downstream
        neural network from a fine-tuning checkpoint, then returns a predictor
        that can run inference directly on raw tabular input data.
        
        Args:
            save_path (str): Path to the fine-tuned model checkpoint file.
            device (torch.device): Device used for model inference.
        
        Returns:
            TabularBERTPredictor: Predictor containing the loaded downstream
                                  model and fitted discretizer.
        
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
            ValueError: If the model configuration is incompatible.
        """
        
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint file not found: {save_path}")
        
        config = torch.load(save_path)
        
        # Load discretizer
        discretizer = UniformDiscretize(
            config['data']['num_bins'], 
            config['data']['encoding_info']
        )
        discretizer.bins = config['discretizer']['bins']
        discretizer.category_maps = config['discretizer']['category_maps']
        
        # Load the pretrained model configuration
        pretrained = TabularBERT(**config['model_config']['tabular_bert'])
        head = MLP(**config['model_config']['mlp_head'])
        model = DownstreamModel(pretrained, head)
        model.load_state_dict(config['model_state_dict'])
        
        return cls(
            model=model,
            discretizer=discretizer,
            device=device
        )
        

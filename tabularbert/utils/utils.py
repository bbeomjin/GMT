import torch
import os
import warnings
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("WandB is not installed. Falling back to TensorBoard for logging. To enable WandB support, install it with: pip install wandb")

class CheckPoint:
    def __init__(self, save_path: str, phase: str, max: bool):
        self.phase = phase
        self.max = max
        self.loss = None
        
        save_path = os.path.expanduser(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        self.save_path = os.path.join(save_path, 'model_checkpoint.pt')
    
    def _create_pretraining_checkpoint(self, discretizer, model, config):
        """Create checkpoint for pretraining phase."""
        return {
            'discretizer': {
                'bins': discretizer.bins,
                'category_maps': discretizer.category_maps
            },
            'data_config': {
                'num_bins': config['data']['num_bins'],
                'encoding_info': config['data']['encoding_info']
            },
            'model_state_dict': model.state_dict(),
            'model_config': {
                'encoding_info': model.encoding_info,
                'embedding_dim': model.embedding_dim,
                'n_layers': model.n_layers,
                'n_heads': model.n_heads,
                'dropout': model.dropout,
                'mode': model.mode
            },
            'regularization_lambda': config['pretraining']['training']['regularization_lambda'],
            'penalty': config['pretraining']['training']['penalty']
        }
    
    def _create_finetuning_checkpoint(self, discretizer, model, config):
        """Create checkpoint for fine-tuning phase."""
        return {
            'discretizer': {
                'bins': discretizer.bins,
                'category_maps': discretizer.category_maps
            },
            'data_config': {
                'num_bins': config['data']['num_bins'],
                'encoding_info': config['data']['encoding_info']
            },
            'model_state_dict': model.state_dict(),
            'model_config': {
                'tabular_bert': {
                    'encoding_info': model.encoding_info,
                    'embedding_dim': model.pretrained.embedding_dim,
                    'n_layers': model.pretrained.n_layers,
                    'n_heads': model.pretrained.n_heads,
                    'dropout': model.pretrained.dropout,
                    'mode': model.pretrained.mode
                },
                'mlp_head': {
                    'input_dim': model.head.input_dim,
                    'output_dim': model.head.output_dim,
                    'hidden_layers': model.head.hidden_layers,
                    'activation': model.head.activation,
                    'dropouts': model.head.dropouts,
                    'batch_norm': model.head.batch_norm
                }
            },
            'regularization_lambda': config['fine-tuning']['training']['regularization_lambda'],
            'penalty': config['fine-tuning']['training']['penalty']
        }
    
    def _save_checkpoint(self, discretizer, model, config):
        """Save checkpoint based on phase."""
        if self.phase == 'pretraining':
            checkpoint = self._create_pretraining_checkpoint(discretizer, model, config)
        else:
            checkpoint = self._create_finetuning_checkpoint(discretizer, model, config)
        
        torch.save(checkpoint, self.save_path)
    
    def __call__(self, x, model, config):
        should_save = False
        
        if self.loss is None:
            # First time - always save
            self.loss = x
            should_save = True
        else:
            # Check if we should update based on max/min criteria
            if (self.max and x > self.loss) or (not self.max and x < self.loss):
                self.loss = x
                should_save = True
        
        if should_save:
            self._save_checkpoint(model, config)



def make_save_dir(dir):
    dir = os.path.join(dir, 'version')
    v = 0
    while True:
        path = dir + str(v)
        if os.path.exists(path) is not True:
            os.mkdir(path)
            break
        else:
            v += 1
    return path



class DualLogger:
    """
    Unified logger that supports both TensorBoard and Weights & Biases (WandB).
    
    This logger automatically logs metrics to both platforms when available,
    providing comprehensive experiment tracking and visualization capabilities.
    
    Args:
        log_dir (str): Directory for TensorBoard logs
        project_name (str, optional): WandB project name. Default: "tabular-bert"
        experiment_name (str, optional): Experiment name for WandB run
        config (dict, optional): Configuration dictionary to log
        use_wandb (bool): Whether to use WandB logging. Default: True
    """
    
    def __init__(self, 
                 log_dir: str,
                 project_name: str="tabular-bert",
                 experiment_name: str=None,
                 config: dict=None,
                 use_wandb: bool=True):
        # Initialize TensorBoard logger
        self.tb_logger = SummaryWriter(log_dir)
        
        # Initialize WandB logger if available and requested
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                dir=os.path.dirname(log_dir)
            )
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value to both TensorBoard and WandB."""
        # Log to TensorBoard
        self.tb_logger.add_scalar(tag, value, step)
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars to both platforms."""
        # Log to TensorBoard
        self.tb_logger.add_scalars(main_tag, tag_scalar_dict, step)
        
        # Log to WandB (flatten the nested structure)
        if self.use_wandb:
            wandb_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            wandb.log(wandb_dict, step=step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram to TensorBoard (WandB histograms handled differently)."""
        self.tb_logger.add_histogram(tag, values, step)
        
        if self.use_wandb:
            # WandB handles histograms differently - log as histogram object
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_model_graph(self, model, input_to_model):
        """Log model graph to TensorBoard."""
        try:
            self.tb_logger.add_graph(model, input_to_model)
        except Exception as e:
            warnings.warn(f"Could not log model graph: {e}")
    
    def close(self):
        """Close both loggers."""
        self.tb_logger.close()
        if self.use_wandb:
            wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        


class EarlyStopping:
    def __init__(self, patience: int=10, mode: str='min'):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
    
    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        elif self.mode == 'max':
            if score > self.best_score:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        if self.counter >= self.patience:
            return True
        return False
    
    
    
def separate_decay_params(model, no_decay_names=None):
    """
    Separate model parameters into two groups: 
    one with weight decay applied and one without.

    Args:
        model (torch.nn.Module): The PyTorch model.
        no_decay_names (list[str], optional): 
            List of substrings used to identify parameters 
            that should NOT have weight decay applied.
            Defaults to ["bias", "layernorm.weight", 'norm1.weight', 'norm2.weight', 'embedding.weight']

    Returns:
        (list, list): A tuple containing:
            - decay_params: parameters with weight decay applied
            - no_decay_params: parameters without weight decay
    """
    if no_decay_names is None:
        no_decay_names = ["bias", "layernorm.weight", 'norm1.weight', 'norm2.weight', 'embedding.weight']

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen parameters
        if any(nd in name for nd in no_decay_names):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay
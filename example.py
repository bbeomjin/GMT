# %%
# TabularBERT quickstart

from pathlib import Path
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from tabularbert import TabularBERTTrainer
from tabularbert.utils.metrics import ClassificationError, RMSE


# %%
# Reproducibility and device setup

def seed_everything(seed: int=0) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_everything(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# %%
# Small helpers used by both examples

def split_data(x, y, stratify=None):
    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        train_size=0.8,
        random_state=0,
        stratify=stratify
    )
    valid_stratify = train_y if stratify is not None else None
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x,
        train_y,
        train_size=0.8,
        random_state=0,
        stratify=valid_stratify
    )
    return train_x, valid_x, test_x, train_y, valid_y, test_y


def transform_features(train_x, valid_x, test_x):
    scaler = QuantileTransformer(
        n_quantiles=min(100, train_x.shape[0]),
        output_distribution='uniform',
        subsample=int(1e9),
        random_state=0
    )
    scaler.fit(train_x)
    return scaler.transform(train_x), scaler.transform(valid_x), scaler.transform(test_x)


def checkpoint_path(save_dir):
    return Path(save_dir) / 'model_checkpoint.pt'


# %%
# Classification example: breast cancer dataset
#
# The target is binary, so the fine-tuning head has output_dim=2.

x_cls, y_cls = load_breast_cancer(return_X_y=True)
train_x_cls, valid_x_cls, test_x_cls, train_y_cls, valid_y_cls, test_y_cls = split_data(
    x_cls,
    y_cls,
    stratify=y_cls
)
train_x_cls, valid_x_cls, test_x_cls = transform_features(
    train_x_cls,
    valid_x_cls,
    test_x_cls
)

print('Classification shapes:')
print('  train_x:', train_x_cls.shape)
print('  valid_x:', valid_x_cls.shape)
print('  test_x :', test_x_cls.shape)


# %%
# Classification: initialize and pretrain TabularBERT

cls_trainer = TabularBERTTrainer(
    x=train_x_cls,
    num_bins=32,
    encoding_info=None,
    device=device
)

# setup_directories_and_logging() expects save_dir itself to already exist.
# Create ./example_runs/breast_cancer/pretraining before running this cell.
cls_trainer.setup_directories_and_logging(
    save_dir='./example_runs/breast_cancer/pretraining',
    phase='pretraining',
    project_name='breast cancer example pretraining',
    use_wandb=False
)
cls_trainer.set_bert(
    embedding_dim=64,
    n_layers=2,
    n_heads=4,
    dropout=0.1,
    mode='add'
)
cls_trainer.set_optimizer(lr=2e-4, weight_decay=1e-5)

cls_trainer.pretrain(
    lamb=0.1,
    epochs=2,
    batch_size=64,
    penalty='squaredL2',
    num_workers=0
)

cls_pretrained_path = checkpoint_path(cls_trainer.save_dir)
print('Classification pretrained checkpoint:', cls_pretrained_path)


# %%
# Classification: load pretrained weights and fine-tune

cls_trainer = TabularBERTTrainer.from_pretrained(
    save_path=cls_pretrained_path,
    device=device
)

# setup_directories_and_logging() expects save_dir itself to already exist.
# Create ./example_runs/breast_cancer/fine-tuning before running this cell.
cls_trainer.setup_directories_and_logging(
    save_dir='./example_runs/breast_cancer/fine-tuning',
    phase='fine-tuning',
    project_name='breast cancer example fine-tuning',
    use_wandb=False
)
cls_trainer.set_head(
    output_dim=2,
    hidden_layers=[64],
    activation='ReLU',
    dropouts=0.1
)
cls_trainer.set_optimizer(lr=1e-4, weight_decay=1e-5)

cls_trainer.finetune(
    x=train_x_cls,
    y=train_y_cls,
    valid_x=valid_x_cls,
    valid_y=valid_y_cls,
    epochs=5,
    batch_size=64,
    penalty='squaredL2',
    criterion=nn.CrossEntropyLoss(),
    metric=ClassificationError(),
    patience=None,
    num_workers=0
)

cls_finetuned_path = checkpoint_path(cls_trainer.save_dir)
print('Classification fine-tuned checkpoint:', cls_finetuned_path)


# %%
# Classification: predict raw tabular data with the fitted discretizer

cls_predictor = TabularBERTTrainer.from_finetuned(
    save_path=cls_finetuned_path,
    device=device
)
cls_logits = cls_predictor.predict(test_x_cls)
cls_pred = cls_logits.argmax(axis=-1)
cls_accuracy = accuracy_score(test_y_cls, cls_pred)

print('Classification logits shape:', cls_logits.shape)
print(f'Breast cancer classification accuracy: {cls_accuracy:.4f}')


# %%
# Regression example: diabetes dataset
#
# The target is continuous. We standardize it for training and transform model
# predictions back to the original target scale for evaluation.

x_reg, y_reg = load_diabetes(return_X_y=True)
train_x_reg, valid_x_reg, test_x_reg, train_y_reg, valid_y_reg, test_y_reg = split_data(
    x_reg,
    y_reg
)
train_x_reg, valid_x_reg, test_x_reg = transform_features(
    train_x_reg,
    valid_x_reg,
    test_x_reg
)

y_mean = train_y_reg.mean()
y_sd = train_y_reg.std()
train_y_reg_scaled = ((train_y_reg - y_mean) / y_sd)[:, None]
valid_y_reg_scaled = ((valid_y_reg - y_mean) / y_sd)[:, None]

print('Regression shapes:')
print('  train_x:', train_x_reg.shape)
print('  valid_x:', valid_x_reg.shape)
print('  test_x :', test_x_reg.shape)
print(f'  target mean/sd: {y_mean:.4f}, {y_sd:.4f}')


# %%
# Regression: initialize and pretrain TabularBERT

reg_trainer = TabularBERTTrainer(
    x=train_x_reg,
    num_bins=32,
    encoding_info=None,
    device=device
)

# setup_directories_and_logging() expects save_dir itself to already exist.
# Create ./example_runs/diabetes/pretraining before running this cell.
reg_trainer.setup_directories_and_logging(
    save_dir='./example_runs/diabetes/pretraining',
    phase='pretraining',
    project_name='diabetes example pretraining',
    use_wandb=False
)
reg_trainer.set_bert(
    embedding_dim=64,
    n_layers=2,
    n_heads=4,
    dropout=0.1,
    mode='add'
)
reg_trainer.set_optimizer(lr=2e-4, weight_decay=1e-5)

reg_trainer.pretrain(
    lamb=0.1,
    epochs=2,
    batch_size=64,
    penalty='squaredL2',
    num_workers=0
)

reg_pretrained_path = checkpoint_path(reg_trainer.save_dir)
print('Regression pretrained checkpoint:', reg_pretrained_path)


# %%
# Regression: load pretrained weights and fine-tune

reg_trainer = TabularBERTTrainer.from_pretrained(
    save_path=reg_pretrained_path,
    device=device
)

# setup_directories_and_logging() expects save_dir itself to already exist.
# Create ./example_runs/diabetes/fine-tuning before running this cell.
reg_trainer.setup_directories_and_logging(
    save_dir='./example_runs/diabetes/fine-tuning',
    phase='fine-tuning',
    project_name='diabetes example fine-tuning',
    use_wandb=False
)
reg_trainer.set_head(
    output_dim=1,
    hidden_layers=[64],
    activation='ReLU',
    dropouts=0.1
)
reg_trainer.set_optimizer(lr=1e-4, weight_decay=1e-5)

reg_trainer.finetune(
    x=train_x_reg,
    y=train_y_reg_scaled,
    valid_x=valid_x_reg,
    valid_y=valid_y_reg_scaled,
    epochs=5,
    batch_size=64,
    penalty='squaredL2',
    criterion=nn.MSELoss(),
    metric=RMSE(weight=y_sd),
    patience=None,
    num_workers=0
)

reg_finetuned_path = checkpoint_path(reg_trainer.save_dir)
print('Regression fine-tuned checkpoint:', reg_finetuned_path)


# %%
# Regression: predict and evaluate on the original target scale

reg_predictor = TabularBERTTrainer.from_finetuned(
    save_path=reg_finetuned_path,
    device=device
)
reg_pred_scaled = reg_predictor.predict(test_x_reg).reshape(-1)
reg_pred = reg_pred_scaled * y_sd + y_mean
reg_rmse = np.sqrt(np.mean((reg_pred - test_y_reg) ** 2))

print('Regression prediction shape:', reg_pred.shape)
print(f'Diabetes regression RMSE: {reg_rmse:.4f}')

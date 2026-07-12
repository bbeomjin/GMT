import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tabularbert import TabularBERTTrainer
from tabularbert.utils.data import prepare_benchmark_data
from tabularbert.utils.metrics import ClassificationError, RMSE


def seed_everything(seed: int=0) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_config(config_path: str | Path) -> tuple[Dict[str, Any], Path]:
    config_path = Path(config_path).resolve()
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config, config_path


def _checkpoint_path(save_dir: str | Path) -> Path:
    return Path(save_dir) / 'model_checkpoint.pt'


def _make_metric(task_type: str, data: Dict[str, Any]):
    if task_type == 'classification':
        return ClassificationError()
    return RMSE(weight=data['target_sd'])


def _make_criterion(task_type: str):
    if task_type == 'classification':
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


def _evaluate(task_type: str, predictions: np.ndarray, labels: np.ndarray, data: Dict[str, Any]) -> Dict[str, float]:
    if task_type == 'classification':
        accuracy = float((predictions.argmax(axis=-1) == labels.reshape(-1)).mean())
        return {'accuracy': accuracy}

    rmse = float(data['target_sd'] * np.sqrt(np.mean((predictions - labels)**2)))
    return {'rmse': rmse}


def _summarize_repeats(repeat_results: list[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_names = [
        key for key in repeat_results[0].keys()
        if key not in {'repeat', 'seed', 'checkpoint'}
    ]
    summary = {}
    for metric_name in metric_names:
        values = np.array([result[metric_name] for result in repeat_results], dtype=float)
        summary[metric_name] = {
            'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'min': float(values.min()),
            'max': float(values.max())
        }
    return summary


def _save_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def run_benchmark(config_path: str | Path) -> Dict[str, Any]:
    config, config_path = _load_config(config_path)
    config_dir = config_path.parent

    dataset_name = config.get('dataset_name', config_dir.name)
    task_type = config['task_type']
    data_dir = _resolve_path(config['data_path'], EXPERIMENTS_DIR)
    output_dir = _resolve_path(
        config.get('output_dir', Path('results') / dataset_name),
        EXPERIMENTS_DIR
    )
    pretraining_dir = output_dir / 'pretraining'
    fine_tuning_dir = output_dir / 'fine-tuning'
    pretraining_dir.mkdir(parents=True, exist_ok=True)
    fine_tuning_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.get('seed', 0))
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    data = prepare_benchmark_data(data_dir, task_type=task_type)

    batch_size = config.get('batch_size', 512)
    pretraining_batch_size = config.get('pretraining_batch_size', batch_size)
    fine_tuning_batch_size = config.get('fine_tuning_batch_size', batch_size)
    fine_tuning_repeats = config.get('fine_tuning_repeats', 20)
    if fine_tuning_repeats < 1:
        raise ValueError('fine_tuning_repeats must be at least 1.')

    trainer = TabularBERTTrainer(
        x=data['train_x'],
        num_bins=config.get('num_bins', 50),
        encoding_info=data['encoding_info'],
        device=device
    )
    trainer.setup_directories_and_logging(
        save_dir=str(pretraining_dir),
        phase='pretraining',
        project_name=f'{dataset_name} data pretraining',
        use_wandb=config.get('use_wandb', False)
    )
    trainer.set_bert(
        embedding_dim=config['embedding_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout=config['pretraining_dropout'],
        mode=config['mode']
    )
    trainer.set_optimizer(
        lr=config.get('pretraining_lr', 2e-4),
        weight_decay=config['weight_decay']
    )
    trainer.pretrain(
        lamb=config['lamb'],
        penalty=config.get('pretraining_penalty', 'squaredL2'),
        epochs=config.get('pretraining_epochs', 1000),
        batch_size=pretraining_batch_size,
        mask_token_prob=config.get('mask_token_prob', 0.15),
        random_token_prob=config.get('random_token_prob', 0.1),
        unchanged_token_prob=config.get('unchanged_token_prob', 0.1),
        num_workers=config.get('num_workers', 0)
    )
    pretrained_checkpoint = _checkpoint_path(trainer.save_dir)

    repeat_results = []
    metrics_path = output_dir / 'metrics.json'
    for repeat in range(fine_tuning_repeats):
        repeat_seed = config.get('seed', 0) + repeat
        repeat_dir = fine_tuning_dir / f'repeat_{repeat:02d}'
        repeat_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(repeat_seed)
        trainer = TabularBERTTrainer.from_pretrained(
            save_path=str(pretrained_checkpoint),
            device=device
        )
        trainer.setup_directories_and_logging(
            save_dir=str(repeat_dir),
            phase='fine-tuning',
            project_name=f'{dataset_name} data fine-tuning repeat {repeat:02d}',
            use_wandb=config.get('use_wandb', False)
        )
        trainer.set_head(
            output_dim=config['output_dim'],
            activation=config.get('activation', 'ReLU'),
            dropouts=config['fine_tuning_dropout'],
            hidden_layers=[config['embedding_dim']] * config['head_n_hidden_layers']
        )
        trainer.set_optimizer(
            lr=config['fine_tuning_lr'],
            weight_decay=config['weight_decay']
        )
        trainer.finetune(
            x=data['train_x'],
            y=data['train_labels'],
            valid_x=data['valid_x'],
            valid_y=data['valid_labels'],
            epochs=config.get('fine_tuning_epochs', 2000),
            penalty=config['fine_tuning_penalty'],
            batch_size=fine_tuning_batch_size,
            criterion=_make_criterion(task_type),
            metric=_make_metric(task_type, data),
            patience=config.get('patience', 100),
            num_workers=config.get('num_workers', 0)
        )
        finetuned_checkpoint = _checkpoint_path(trainer.save_dir)

        predictor = TabularBERTTrainer.from_finetuned(
            save_path=str(finetuned_checkpoint),
            device=device
        )
        predictions = predictor.predict(
            data['test_x'],
            batch_size=config.get('prediction_batch_size', 4096)
        )
        metrics = _evaluate(task_type, predictions, data['test_labels'], data)
        repeat_result = {
            'repeat': repeat,
            'seed': repeat_seed,
            'checkpoint': str(finetuned_checkpoint),
            **metrics
        }
        repeat_results.append(repeat_result)

        _save_json(repeat_result, repeat_dir / 'metrics.json')
        results = {
            'dataset_name': dataset_name,
            'task_type': task_type,
            'pretrained_checkpoint': str(pretrained_checkpoint),
            'fine_tuning_repeats': fine_tuning_repeats,
            'completed_repeats': len(repeat_results),
            'repeats': repeat_results,
            'summary': _summarize_repeats(repeat_results)
        }
        _save_json(results, metrics_path)
        print(f'{dataset_name} repeat {repeat:02d} results: {metrics}')

    results = {
        'dataset_name': dataset_name,
        'task_type': task_type,
        'pretrained_checkpoint': str(pretrained_checkpoint),
        'fine_tuning_repeats': fine_tuning_repeats,
        'completed_repeats': len(repeat_results),
        'repeats': repeat_results,
        'summary': _summarize_repeats(repeat_results)
    }

    _save_json(results, metrics_path)

    print(f'{dataset_name} summary: {results["summary"]}')
    print(f'Checkpoints and metrics saved to: {output_dir}')
    return results


def main(config_path: str | Path | None=None) -> Dict[str, Any]:
    if config_path is None:
        parser = argparse.ArgumentParser(description='Run a TabularBERT benchmark experiment.')
        parser.add_argument('--config', required=True, help='Path to a benchmark config JSON file.')
        args = parser.parse_args()
        config_path = args.config

    return run_benchmark(config_path)


if __name__ == '__main__':
    main()

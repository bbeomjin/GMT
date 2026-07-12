# [ICML 2026] TabularBERT: Binning-Based Self-Supervised Learning for Tabular Representation

This repository provides the official implementation of **TabularBERT**, a Transformer-based model for tabular data that tokenizes continuous variables via binning and learns numerically structured representations through MLM pretraining and downstream fine-tuning.

<p align="center">
<img src="./TabularBERT_pretraining.png" width="800">
</p>

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/bbeomjin/tabularbert.git
cd tabularbert
pip install -e .
```

The required packages are listed in `requirements.txt`


## Datasets

This repository does not include or redistribute the benchmark datasets.

To reproduce our experiments, please download the corresponding processed datasets from the following repository:
- https://github.com/jyansir/t2g-former

Users are responsible for complying with the licenses and terms of the original dataset providers.


## Quick Start

Here's a basic example of how to use TabularBERT:

```python
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from tabularbert import TabularBERTTrainer, TabularBERTPredictor
from tabularbert.utils.metrics import ClassificationError, RMSE

# Set task: "classification" or "regression"
task = "classification"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your tabular data
data = pd.read_csv("your_dataset.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=0)

# Scale features
scaler = QuantileTransformer(n_quantiles=max(min(train_X.shape[0] // 30, 1000), 10),
                             output_distribution='uniform')
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

# Initialize TabularBERT trainer
trainer = TabularBERTTrainer(
    x=train_X_scaled,
    num_bins=50,
    device=device
)

# Setup directories and logging for pretraining
trainer.setup_directories_and_logging(
    save_dir='./pretraining',
    phase='pretraining',
    project_name='My TabularBERT Project'
)

# Start pretraining
trainer.pretrain()

# Setup directories and logging for fine-tuning
trainer.setup_directories_and_logging(
    save_dir='./fine-tuning',
    phase='fine-tuning',
    project_name='My TabularBERT Project'
)

# Start fine-tuning
trainer.finetune(
   x=train_X_scaled,
   y=train_y,
   criterion=torch.nn.CrossEntropyLoss() if task == 'classification' else torch.nn.MSELoss(),
   metric=ClassificationError() if task == 'classification' else RMSE()
)

# Prediction
predictor = TabularBERTPredictor(model=trainer.model, discretizer=trainer.discretizer, device=device)
predictions = predictor.predict(test_X_scaled)
```

- For more detailed documentation and advanced usage examples, please refer to: `example.py`

## Citation

If you use TabularBERT in your research, please cite:

```bibtex
@inproceedings{park2026tabularbert,
    title={TabularBERT: Binning-Based Self-Supervised Learning for Tabular Representation},
    author={Beomjin Park and Seunghwan An and Sungchul Hong and Hosik Choi},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

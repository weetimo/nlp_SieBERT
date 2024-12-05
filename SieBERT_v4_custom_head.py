#to run: python /home/user/Documents/Tim/NLP/final_project/tim_q4/design_challenge_tim_vanilla.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AdamW,
    get_linear_schedule_with_warmup
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import random
import os
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import yaml
from sklearn.model_selection import KFold
from utils.data_augmentation import TextAugmenter
from utils.metrics import MetricsTracker
from utils.training import EarlyStopping, ModelCheckpointing

# Load configuration
import os
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SentimentDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
        augment: bool = False
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if augment and CONFIG['data']['augmentation']['enabled']:
            self.augmenter = TextAugmenter(
                synonym_prob=CONFIG['data']['augmentation']['synonym_replacement_prob'],
                deletion_prob=CONFIG['data']['augmentation']['random_deletion_prob'],
                max_aug_per_sample=CONFIG['data']['augmentation']['max_aug_per_sample']
            )
            augmented_texts = []
            augmented_labels = []
            for text, label in zip(texts, labels):
                aug_texts = self.augmenter.augment(text)
                augmented_texts.extend(aug_texts)
                augmented_labels.extend([label] * len(aug_texts))
            
            self.texts.extend(augmented_texts)
            self.labels.extend(augmented_labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text  # Add text for error analysis
        }

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int = CONFIG['model']['num_classes']):
        super().__init__()
        self.siebert = AutoModel.from_pretrained(CONFIG['model']['name'])
        hidden_size = self.siebert.config.hidden_size  # Dynamically get hidden size
        
        # Custom head architecture
        self.drop = nn.Dropout(p=CONFIG['model']['dropout_rate'])
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
        
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(256)

    def forward(self, input_ids, attention_mask):
        outputs = self.siebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply custom head
        x = self.drop(pooled_output)
        
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.drop(x)
        
        x = self.fc3(x)
        return x

class SentimentTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        n_classes: int = CONFIG['model']['num_classes'],
        max_length: int = CONFIG['model']['max_length']
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_classes = n_classes
        self.max_length = max_length
        self.criterion = nn.CrossEntropyLoss()
        self.metrics_tracker = MetricsTracker(
            log_misclassified=CONFIG['logging']['log_misclassified']
        )
        logger.info(f"Initialized trainer with device: {device}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        run_dir: str,
        timestamp: str,
        epochs: int = CONFIG['training']['epochs'],
        max_grad_norm: float = CONFIG['training']['max_grad_norm'],
        gradient_accumulation_steps: int = CONFIG['training']['gradient_accumulation_steps']
    ):
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=CONFIG['training']['early_stopping']['patience'],
            min_delta=CONFIG['training']['early_stopping']['min_delta'],
            mode='max'  # monitoring f1 score
        )
        
        # Initialize model checkpointing
        checkpointer = ModelCheckpointing(
            save_dir=run_dir,
            save_top_k=CONFIG['logging']['save_top_k'],
            metrics_to_track=CONFIG['logging']['metrics_to_track']
        )

        # Optimizer setup with gradient checkpointing
        if CONFIG['model']['gradient_checkpointing']:
            self.model.siebert.gradient_checkpointing_enable()
        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': CONFIG['training']['weight_decay']
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(
            [
                {
                    'params': optimizer_grouped_parameters[0]['params'],
                    'weight_decay': CONFIG['training']['weight_decay'],
                    'lr': float(CONFIG['training']['learning_rate'])  # Convert to float
                },
                {
                    'params': optimizer_grouped_parameters[1]['params'],
                    'weight_decay': 0.0,
                    'lr': float(CONFIG['training']['learning_rate'])  # Convert to float
                }
            ]
        )
        
        # Learning rate scheduler setup
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=2,
            verbose=True
        )
        warmup_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CONFIG['training']['warmup_steps'],
            num_training_steps=len(train_loader) * epochs // gradient_accumulation_steps
        )

        scaler = torch.cuda.amp.GradScaler()
        global_step = 0
        
        try:
            for epoch in range(epochs):
                logger.info(f"Starting epoch {epoch + 1}/{epochs}")
                self.model.train()
                total_loss = 0
                optimizer.zero_grad()

                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                    try:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        with torch.cuda.amp.autocast():
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                            loss = self.criterion(outputs, labels)
                            loss = loss / gradient_accumulation_steps

                        scaler.scale(loss).backward()
                        total_loss += loss.item()

                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_grad_norm
                            )
                            scaler.step(optimizer)
                            scaler.update()
                            warmup_scheduler.step()
                            optimizer.zero_grad()
                            global_step += 1
                        
                        if batch_idx % CONFIG['logging']['log_interval'] == 0:
                            step_loss = loss.item() * gradient_accumulation_steps
                            logger.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {step_loss:.4f}")
                            wandb.log({"batch_loss": step_loss}, step=global_step)

                    except Exception as e:
                        logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                        continue

                avg_train_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
                
                # Validation phase
                logger.info("Running validation...")
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Validation Metrics: {val_metrics}")
                
                # Update learning rate based on validation f1
                lr_scheduler.step(val_metrics['f1'])
                
                # Log metrics
                metrics_dict = {
                    "train_loss": avg_train_loss,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                
                for metric_name, metric_value in val_metrics.items():
                    metrics_dict[f"val_{metric_name}"] = metric_value

                wandb.log(metrics_dict, step=global_step)
                
                # Model checkpointing
                checkpointer.save_checkpoint(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    metrics=val_metrics,
                    epoch=epoch,
                    global_step=global_step,
                    timestamp=timestamp
                )
                
                # Early stopping check
                if early_stopping(val_metrics['f1']):
                    logger.info("Early stopping triggered")
                    break

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise e

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        logger.info(f"Evaluating on {len(data_loader.dataset)} samples")
        self.model.eval()
        predictions = []
        actual_labels = []
        total_loss = 0
        texts = []

        try:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Evaluating"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    texts.extend(batch['text'])

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    
                    _, preds = torch.max(outputs, dim=1)
                    
                    predictions.extend(preds.cpu().numpy())
                    actual_labels.extend(labels.cpu().numpy())

            metrics = self.metrics_tracker.compute_metrics(
                np.array(predictions),
                np.array(actual_labels),
                texts=texts
            )
            metrics['loss'] = total_loss / len(data_loader)
            
            return metrics

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise e

class DataProcessor:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def load_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """Load and process the sentiment dataset."""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, header=None, names=['review'])
        
        # Determine sentiment from filename
        is_positive = 'pos' in file_path.lower()
        labels = [1 if is_positive else 0] * len(df)
        
        # Clean the reviews
        texts = df['review'].tolist()
        
        logger.info(f"Loaded {len(texts)} reviews with {'positive' if is_positive else 'negative'} sentiment")
        return texts, labels
    
    def create_data_splits(
        self,
        texts: List[str],
        labels: List[int],
        train_ratio: float = CONFIG['data']['train_ratio'],
        val_ratio: float = CONFIG['data']['val_ratio']
    ) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Split data into train, validation, and test sets."""
        logger.info("Creating data splits...")
        total_samples = len(texts)
        indices = np.random.permutation(total_samples)
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        splits = {
            'train': {
                'texts': [texts[i] for i in train_indices],
                'labels': [labels[i] for i in train_indices]
            },
            'val': {
                'texts': [texts[i] for i in val_indices],
                'labels': [labels[i] for i in val_indices]
            },
            'test': {
                'texts': [texts[i] for i in test_indices],
                'labels': [labels[i] for i in test_indices]
            }
        }
        
        logger.info(f"Split sizes - Train: {len(splits['train']['texts'])}, "
                   f"Val: {len(splits['val']['texts'])}, "
                   f"Test: {len(splits['test']['texts'])}")
        return splits

def create_timestamped_dir():
    """Create a timestamped directory for the current training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract model name from path
    model_name = CONFIG['model']['name'].split('/')[-1]
    run_dir = os.path.join(CONFIG['paths']['output_dir'], f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp

def write_model_metrics_readme(run_dir: str, timestamp: str, best_metrics: Dict, final_metrics: Dict):
    """Write a README file with model metrics."""
    model_name = CONFIG['model']['name'].split('/')[-1]
    
    readme_content = f"""# Training Run Results - {timestamp}
Model: {model_name}
Current Epoch: {best_metrics['epoch']} / {best_metrics['total_epochs']}

## Best Model Metrics (best_model_{timestamp}.pt)
### Validation Metrics:
{json.dumps(best_metrics['validation_metrics'], indent=2)}

### Test Metrics:
{json.dumps(best_metrics['test_metrics'], indent=2)}

### Training Info:
- Epoch: {best_metrics['epoch']}
- Training Loss: {best_metrics['train_loss']}

"""
    # Only add final metrics if they're different from best metrics
    if final_metrics != best_metrics:
        readme_content += f"""## Final Model Metrics (final_model_{timestamp}.pt)
### Validation Metrics:
{json.dumps(final_metrics['validation_metrics'], indent=2)}

### Test Metrics:
{json.dumps(final_metrics['test_metrics'], indent=2)}

### Training Info:
- Total Epochs: {final_metrics['total_epochs']}
"""
    
    readme_path = os.path.join(run_dir, f"README_{timestamp}.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)

def main():
    logger.info("Starting sentiment analysis training")
    
    # Create timestamped directory for this run
    run_dir, timestamp = create_timestamped_dir()
    logger.info(f"Created run directory: {run_dir}")
    
    # Initialize wandb for experiment tracking
    wandb.init(project="NLP_SieBERT", config=CONFIG)
    logger.info("Initialized wandb")
    
    # Set device and random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(CONFIG['training']['seed'])
    logger.info(f"Using device: {device}")
    
    # Load and process data
    data_processor = DataProcessor()
    
    # Load all data
    logger.info("Loading datasets...")
    train_pos_texts, train_pos_labels = data_processor.load_data(CONFIG['paths']['train_pos_path'])
    train_neg_texts, train_neg_labels = data_processor.load_data(CONFIG['paths']['train_neg_path'])
    test_pos_texts, test_pos_labels = data_processor.load_data(CONFIG['paths']['test_pos_path'])
    test_neg_texts, test_neg_labels = data_processor.load_data(CONFIG['paths']['test_neg_path'])
    
    # Combine datasets
    train_texts = train_pos_texts + train_neg_texts
    train_labels = train_pos_labels + train_neg_labels
    test_texts = test_pos_texts + test_neg_texts
    test_labels = test_pos_labels + test_neg_labels
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model']['name'])
    
    # Create test dataset
    test_dataset = SentimentDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=CONFIG['model']['max_length']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['training']['eval_batch_size'],
        shuffle=False,
        num_workers=CONFIG['data']['num_workers']
    )
    
    if CONFIG['training']['cross_validation']['enabled']:
        # Perform k-fold cross validation
        kf = KFold(
            n_splits=CONFIG['training']['cross_validation']['n_folds'],
            shuffle=True,
            random_state=CONFIG['training']['seed']
        )
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_texts)):
            logger.info(f"Starting fold {fold + 1}")
            
            # Split data for this fold
            fold_train_texts = [train_texts[i] for i in train_idx]
            fold_train_labels = [train_labels[i] for i in train_idx]
            fold_val_texts = [train_texts[i] for i in val_idx]
            fold_val_labels = [train_labels[i] for i in val_idx]
            
            # Create datasets for this fold
            train_dataset = SentimentDataset(
                texts=fold_train_texts,
                labels=fold_train_labels,
                tokenizer=tokenizer,
                max_length=CONFIG['model']['max_length'],
                augment=True
            )
            val_dataset = SentimentDataset(
                texts=fold_val_texts,
                labels=fold_val_labels,
                tokenizer=tokenizer,
                max_length=CONFIG['model']['max_length']
            )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=CONFIG['training']['train_batch_size'],
                shuffle=True,
                num_workers=CONFIG['data']['num_workers']
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=CONFIG['training']['eval_batch_size'],
                shuffle=False,
                num_workers=CONFIG['data']['num_workers']
            )
            
            # Initialize model for this fold
            model = SentimentClassifier()
            model = model.to(device)
            
            # Initialize trainer
            trainer = SentimentTrainer(
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            # Train the model
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                run_dir=os.path.join(run_dir, f'fold_{fold}'),
                timestamp=f"{timestamp}_fold_{fold}",
                epochs=CONFIG['training']['epochs']
            )
            
            # Get final metrics for this fold
            fold_val_metrics = trainer.evaluate(val_loader)
            fold_metrics.append(fold_val_metrics)
            
            # Log fold metrics
            wandb.log({
                f"fold_{fold}_metrics": fold_val_metrics
            })
        
        # Calculate and log average metrics across folds
        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            avg_metrics[metric] = sum(fold[metric] for fold in fold_metrics) / len(fold_metrics)
        
        wandb.log({
            "average_cv_metrics": avg_metrics
        })
        
        logger.info("Cross-validation completed")
        logger.info(f"Average metrics across folds: {avg_metrics}")
    
    else:
        # Regular training without cross-validation
        train_val_splits = data_processor.create_data_splits(
            texts=train_texts,
            labels=train_labels
        )
        
        # Create datasets
        train_dataset = SentimentDataset(
            texts=train_val_splits['train']['texts'],
            labels=train_val_splits['train']['labels'],
            tokenizer=tokenizer,
            max_length=CONFIG['model']['max_length'],
            augment=True
        )
        val_dataset = SentimentDataset(
            texts=train_val_splits['val']['texts'],
            labels=train_val_splits['val']['labels'],
            tokenizer=tokenizer,
            max_length=CONFIG['model']['max_length']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['training']['train_batch_size'],
            shuffle=True,
            num_workers=CONFIG['data']['num_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['training']['eval_batch_size'],
            shuffle=False,
            num_workers=CONFIG['data']['num_workers']
        )
        
        # Initialize model
        model = SentimentClassifier()
        model = model.to(device)
        
        # Initialize trainer
        trainer = SentimentTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Train the model
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_dir=run_dir,
            timestamp=timestamp,
            epochs=CONFIG['training']['epochs']
        )
        
        # Get final metrics
        final_val_metrics = trainer.evaluate(val_loader)
        final_test_metrics = trainer.evaluate(test_loader)
        
        # Log final metrics
        wandb.log({
            "final_metrics": {
                "validation": final_val_metrics,
                "test": final_test_metrics
            }
        })
        
        logger.info("Training completed")
        logger.info(f"Final validation metrics: {final_val_metrics}")
        logger.info(f"Final test metrics: {final_test_metrics}")
    
    wandb.finish()
    logger.info(f"All results saved in: {run_dir}")

if __name__ == "__main__":
    main()
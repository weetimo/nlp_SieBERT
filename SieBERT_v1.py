#to run: python /home/user/Documents/Tim/NLP/final_project/tim_q4/design_challenge_tim_vanilla.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import random
import os
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import json

# Hyperparameter Configuration
CONFIG = {
    # Model Configuration
    'model_name': 'siebert/sentiment-roberta-large-english',
    'num_classes': 2,
    'max_length': 128,
    'dropout_rate': 0.3,
    
    # Training Configuration
    'seed': 42,
    'epochs': 100,
    'train_batch_size': 16,
    'eval_batch_size': 32,
    'learning_rate': 2e-5,
    'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 4,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    
    # Data Configuration
    'train_ratio': 0.9,
    'val_ratio': 0.1,
    'num_workers': 4,
    
    # Paths Configuration
    'base_path': '/home/user/Documents/Tim/NLP/final_project/tim_q4',
    'train_pos_path': '/home/user/Documents/Tim/NLP/final_project/tim_q4/data/train/mini_train_pos.csv',
    'train_neg_path': '/home/user/Documents/Tim/NLP/final_project/tim_q4/data/train/mini_train_neg.csv',
    'test_pos_path': '/home/user/Documents/Tim/NLP/final_project/tim_q4/data/testing/mini_test_pos.csv',
    'test_neg_path': '/home/user/Documents/Tim/NLP/final_project/tim_q4/data/testing/mini_test_neg.csv',
    'output_dir': '/home/user/Documents/Tim/NLP/final_project/tim_q4/training_runs'
}

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
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int = CONFIG['num_classes']):
        super().__init__()
        self.siebert = AutoModelForSequenceClassification.from_pretrained(
            CONFIG['model_name'],
            num_labels=n_classes
        )
        self.drop = nn.Dropout(p=CONFIG['dropout_rate'])

    def forward(self, input_ids, attention_mask):
        outputs = self.siebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

class SentimentTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        n_classes: int = CONFIG['num_classes'],
        max_length: int = CONFIG['max_length']
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_classes = n_classes
        self.max_length = max_length
        self.criterion = nn.CrossEntropyLoss()
        logger.info(f"Initialized trainer with device: {device}")

    def compute_metrics(self, predictions, labels):
        preds_flat = predictions.flatten()
        labels_flat = labels.flatten()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, 
            preds_flat, 
            average='binary'
        )
        acc = accuracy_score(labels_flat, preds_flat)
        metrics = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        logger.info(f"Metrics: {metrics}")
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        run_dir: str,
        timestamp: str,
        epochs: int = CONFIG['epochs'],
        max_grad_norm: float = CONFIG['max_grad_norm'],
        gradient_accumulation_steps: int = CONFIG['gradient_accumulation_steps']
    ):
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': CONFIG['weight_decay']
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=CONFIG['learning_rate'])
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CONFIG['warmup_steps'],
            num_training_steps=total_steps
        )

        scaler = torch.cuda.amp.GradScaler()
        logger.info(f"Total training steps: {total_steps}")
        
        best_val_f1 = 0
        global_step = 0
        
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                if batch_idx % 100 == 0:
                    step_loss = loss.item() * gradient_accumulation_steps
                    logger.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {step_loss:.4f}")
                    wandb.log({"batch_loss": step_loss}, step=global_step)

            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
            
            metrics_dict = {
                "train_loss": avg_train_loss,
            }

            logger.info("Running validation...")
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Validation Metrics: {val_metrics}")
            
            for metric_name, metric_value in val_metrics.items():
                metrics_dict[f"val_{metric_name}"] = metric_value

            logger.info("Running test evaluation...")
            test_metrics = self.evaluate(test_loader)
            logger.info(f"Test Metrics: {test_metrics}")
            
            for metric_name, metric_value in test_metrics.items():
                metrics_dict[f"test_{metric_name}"] = metric_value

            wandb.log(metrics_dict, step=global_step)

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                logger.info(f"New best F1 score: {best_val_f1:.4f} - Saving model")
                model_path = os.path.join(run_dir, f'best_model_{timestamp}.pt')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'tokenizer': self.tokenizer,
                    'config': self.model.config if hasattr(self.model, 'config') else None,
                    'epoch': epoch,
                    'validation_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'train_loss': avg_train_loss,
                    'global_step': global_step
                }, model_path)

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        logger.info(f"Evaluating on {len(data_loader.dataset)} samples")
        self.model.eval()
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())

        metrics = self.compute_metrics(
            np.array(predictions),
            np.array(actual_labels)
        )
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

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
        train_ratio: float = CONFIG['train_ratio'],
        val_ratio: float = CONFIG['val_ratio']
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
    run_dir = os.path.join(CONFIG['output_dir'], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp

def write_model_metrics_readme(run_dir: str, timestamp: str, best_metrics: Dict, final_metrics: Dict):
    """Write a README file with model metrics."""
    readme_content = f"""# Training Run Results - {timestamp}

## Best Model Metrics (best_model_{timestamp}.pt)
### Validation Metrics:
{json.dumps(best_metrics['validation_metrics'], indent=2)}

### Test Metrics:
{json.dumps(best_metrics['test_metrics'], indent=2)}

### Training Info:
- Epoch: {best_metrics['epoch']}
- Training Loss: {best_metrics['train_loss']}

## Final Model Metrics (final_model_{timestamp}.pt)
### Validation Metrics:
{json.dumps(final_metrics['final_validation_metrics'], indent=2)}

### Test Metrics:
{json.dumps(final_metrics['final_test_metrics'], indent=2)}

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
    set_seed(CONFIG['seed'])
    logger.info(f"Using device: {device}")
    
    # Initialize data processor and load data
    data_processor = DataProcessor()
    
    # Load training data
    logger.info("Loading training datasets...")
    train_pos_texts, train_pos_labels = data_processor.load_data(CONFIG['train_pos_path'])
    train_neg_texts, train_neg_labels = data_processor.load_data(CONFIG['train_neg_path'])
    
    # Load test data
    logger.info("Loading test datasets...")
    test_pos_texts, test_pos_labels = data_processor.load_data(CONFIG['test_pos_path'])
    test_neg_texts, test_neg_labels = data_processor.load_data(CONFIG['test_neg_path'])
    
    # Combine training datasets
    train_texts = train_pos_texts + train_neg_texts
    train_labels = train_pos_labels + train_neg_labels
    logger.info(f"Training dataset size: {len(train_texts)} samples")
    
    # Combine test datasets
    test_texts = test_pos_texts + test_neg_texts
    test_labels = test_pos_labels + test_neg_labels
    logger.info(f"Test dataset size: {len(test_texts)} samples")
    
    # Create validation split from training data
    train_val_splits = data_processor.create_data_splits(
        texts=train_texts,
        labels=train_labels,
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio']
    )
    
    # Initialize tokenizer and model
    logger.info("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = SentimentClassifier()
    model = model.to(device)
    logger.info("Model initialized and moved to device")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = SentimentDataset(
        texts=train_val_splits['train']['texts'],
        labels=train_val_splits['train']['labels'],
        tokenizer=tokenizer
    )
    val_dataset = SentimentDataset(
        texts=train_val_splits['val']['texts'],
        labels=train_val_splits['val']['labels'],
        tokenizer=tokenizer
    )
    test_dataset = SentimentDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer
    )

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['train_batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['eval_batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['eval_batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )

    # Initialize trainer
    trainer = SentimentTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Train the model
    logger.info("Starting training process...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        run_dir=run_dir,
        timestamp=timestamp,
        epochs=CONFIG['epochs']
    )
    
    # Get final metrics
    final_val_metrics = trainer.evaluate(val_loader)
    final_test_metrics = trainer.evaluate(test_loader)
    
    # Save final model with all metrics
    logger.info("Saving final model...")
    final_model_path = os.path.join(run_dir, f'final_model_{timestamp}.pt')
    final_model_data = {
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': model.config if hasattr(model, 'config') else None,
        'final_validation_metrics': final_val_metrics,
        'final_test_metrics': final_test_metrics,
        'total_epochs': CONFIG['epochs']
    }
    torch.save(final_model_data, final_model_path)
    
    # Log final metrics summary
    logger.info("Final Model Metrics:")
    logger.info(f"Validation Metrics: {final_val_metrics}")
    logger.info(f"Test Metrics: {final_test_metrics}")
    
    # Compare with best model metrics
    best_model_path = os.path.join(run_dir, f'best_model_{timestamp}.pt')
    best_model_data = torch.load(best_model_path)
    
    # Write README with metrics
    write_model_metrics_readme(run_dir, timestamp, best_model_data, final_model_data)
    
    logger.info(f"\nBest Model Metrics (from epoch {best_model_data['epoch']}, step {best_model_data['global_step']}):")
    logger.info(f"Validation Metrics: {best_model_data['validation_metrics']}")
    logger.info(f"Test Metrics: {best_model_data['test_metrics']}")
    
    wandb.log({
        "final_model_val_metrics": final_val_metrics,
        "final_model_test_metrics": final_test_metrics,
        "best_model_val_metrics": best_model_data['validation_metrics'],
        "best_model_test_metrics": best_model_data['test_metrics']
    }, step=best_model_data['global_step'] + 1)  
    
    logger.info(f"Training completed. Results saved in: {run_dir}")

if __name__ == "__main__":
    main()
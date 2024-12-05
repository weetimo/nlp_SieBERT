import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, List, Tuple
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsTracker:
    def __init__(self, log_misclassified: bool = True):
        self.log_misclassified = log_misclassified
        self.misclassified = []

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        texts: List[str] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        preds_flat = predictions.flatten()
        labels_flat = labels.flatten()
        
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat,
            preds_flat,
            average='binary'
        )
        acc = accuracy_score(labels_flat, preds_flat)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(labels_flat, preds_flat)
        except ValueError:
            roc_auc = 0.0
        
        # Confusion Matrix
        cm = confusion_matrix(labels_flat, preds_flat)
        
        # Log confusion matrix to wandb
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        metrics = {
            f'{prefix}accuracy': acc,
            f'{prefix}f1': f1,
            f'{prefix}precision': precision,
            f'{prefix}recall': recall,
            f'{prefix}roc_auc': roc_auc,
        }
        
        # Log confusion matrix
        if wandb.run is not None:
            wandb.log({f"{prefix}confusion_matrix": wandb.Image(fig)})
        plt.close()

        # Track misclassified examples
        if self.log_misclassified and texts is not None:
            misclassified_indices = np.where(preds_flat != labels_flat)[0]
            for idx in misclassified_indices:
                self.misclassified.append({
                    'text': texts[idx],
                    'true_label': int(labels_flat[idx]),
                    'predicted_label': int(preds_flat[idx])
                })
            
            # Log sample of misclassified examples to wandb
            if wandb.run is not None and self.misclassified:
                sample_size = min(10, len(self.misclassified))
                wandb.log({
                    f"{prefix}misclassified_examples": wandb.Table(
                        data=[[m['text'], m['true_label'], m['predicted_label']] 
                             for m in self.misclassified[:sample_size]],
                        columns=["Text", "True Label", "Predicted Label"]
                    )
                })

        return metrics

    def log_learning_curves(self, train_metrics: Dict, val_metrics: Dict, step: int):
        """Log learning curves to wandb"""
        if wandb.run is not None:
            metrics_to_log = {}
            for metric in train_metrics:
                metrics_to_log[f'train_{metric}'] = train_metrics[metric]
                if metric in val_metrics:
                    metrics_to_log[f'val_{metric}'] = val_metrics[metric]
            wandb.log(metrics_to_log, step=step)

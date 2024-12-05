import torch
from typing import Dict, Optional
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'max':
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

class ModelCheckpointing:
    def __init__(self, save_dir: str, save_top_k: int = 3, metrics_to_track: list = None):
        self.save_dir = save_dir
        self.save_top_k = save_top_k
        # Only track f1 score by default, as it's the most important metric
        self.metrics_to_track = ['f1'] if metrics_to_track is None else metrics_to_track
        self.best_metrics = {metric: [] for metric in self.metrics_to_track}
        # Save tokenizer only once at initialization
        self.tokenizer_saved = False
        logger.info(f"Initialized ModelCheckpointing with save_dir: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        tokenizer,
        metrics: Dict,
        epoch: int,
        global_step: int,
        timestamp: str,
        is_final: bool = False
    ):
        logger.info(f"Attempting to save checkpoint for epoch {epoch}")
        
        # Save tokenizer only once, in a common location
        if not self.tokenizer_saved:
            tokenizer_dir = os.path.join(self.save_dir, 'tokenizer')
            logger.info(f"Saving tokenizer to: {tokenizer_dir}")
            tokenizer.save_pretrained(tokenizer_dir)
            self.tokenizer_saved = True
            self.tokenizer_dir = tokenizer_dir
        
        # If this is the final model, save it regardless of metrics
        if is_final:
            final_path = os.path.join(self.save_dir, f'model_final_{timestamp}.pt')
            try:
                self._save_model(model, metrics, epoch, global_step, final_path)
                logger.info(f"Successfully saved final model to {final_path}")
            except Exception as e:
                logger.error(f"Error saving final model: {str(e)}")
        
        # Continue with regular checkpoint saving for best models
        for metric_name in self.metrics_to_track:
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                checkpoint_info = {
                    'value': metric_value,
                    'epoch': epoch,
                    'path': os.path.join(
                        self.save_dir,
                        f'model_best_{metric_name}_{timestamp}.pt'
                    )
                }
                logger.info(f"Checking checkpoint for metric {metric_name} with value {metric_value}")

                # Check if this metric value is in top-k
                self.best_metrics[metric_name].append(checkpoint_info)
                self.best_metrics[metric_name].sort(
                    key=lambda x: x['value'],
                    reverse=True
                )

                # Only save if this is actually one of the top-k models
                should_save = False
                if len(self.best_metrics[metric_name]) <= self.save_top_k:
                    should_save = True
                elif checkpoint_info in self.best_metrics[metric_name][:self.save_top_k]:
                    should_save = True
                    # Remove old checkpoint file
                    old_checkpoint = self.best_metrics[metric_name][self.save_top_k]['path']
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                        logger.info(f"Removed old checkpoint: {old_checkpoint}")
                
                if should_save:
                    try:
                        self._save_model(
                            model, metrics, epoch,
                            global_step, checkpoint_info['path']
                        )
                        logger.info(f"Successfully saved checkpoint to {checkpoint_info['path']}")
                    except Exception as e:
                        logger.error(f"Error saving checkpoint: {str(e)}")
                
                self.best_metrics[metric_name] = self.best_metrics[metric_name][:self.save_top_k]

    def _save_model(
        self,
        model: torch.nn.Module,
        metrics: Dict,
        epoch: int,
        global_step: int,
        path: str
    ):
        try:
            # Ensure save directory exists
            save_dir = os.path.dirname(path)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model and metadata
            logger.info(f"Saving model to: {path}")
            model_data = {
                'model_state_dict': model.state_dict(),
                'tokenizer_path': self.tokenizer_dir,  # Reference the common tokenizer
                'config': model.config if hasattr(model, 'config') else None,
                'epoch': epoch,
                'metrics': metrics,
                'global_step': global_step
            }
            torch.save(model_data, path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error in _save_model: {str(e)}")
            raise

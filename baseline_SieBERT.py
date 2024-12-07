import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import Dataset
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_test_data():
    # Load positive and negative test data
    test_pos = pd.read_csv('/home/user/Documents/Tim/NLP/final_project/tim_q4/full_data/testing/test_pos.csv')
    test_neg = pd.read_csv('/home/user/Documents/Tim/NLP/final_project/tim_q4/full_data/testing/test_neg.csv')
    
    # Combine the data
    texts = test_pos['text'].tolist() + test_neg['text'].tolist()
    labels = [1] * len(test_pos) + [0] * len(test_neg)
    
    # Create a dataset dictionary
    dataset_dict = {
        'text': texts,
        'label': labels
    }
    
    # Convert to HuggingFace Dataset
    return Dataset.from_dict(dataset_dict)

def evaluate_predictions(true_labels, predictions):
    # Extract predicted labels and probabilities
    pred_labels = [1 if pred['label'] == 'POSITIVE' else 0 for pred in predictions]
    pred_probs = [pred['score'] if pred['label'] == 'POSITIVE' else 1 - pred['score'] for pred in predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    auc_roc = roc_auc_score(true_labels, pred_probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }

def main():
    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Using {'GPU' if device == 0 else 'CPU'} for inference")
    
    model_name = "siebert/sentiment-roberta-large-english"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize the pipeline with GPU
    sentiment_analysis = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=tokenizer,
        device=device,
        max_length=512,
        truncation=True,
        batch_size=32
    )
    
    # Load test data as a dataset
    logger.info("Loading test data...")
    test_dataset = load_test_data()
    
    # Run predictions using the dataset
    logger.info("Running predictions on test data...")
    predictions = sentiment_analysis(test_dataset['text'])
    
    # Evaluate results
    logger.info("Calculating metrics...")
    metrics = evaluate_predictions(test_dataset['label'], predictions)
    
    # Print results
    logger.info("\nEvaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()

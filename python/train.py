"""
Training script for NanoDeepSeek model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

from nanodeepseek.model.model import NanoDeepSeek
from nanodeepseek.data.tokenizer import build_tokenizer
from  transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for training"""
    
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        self.tokens = []
        for text in texts:
            token_ids = tokenizer.encode(text)
            
            # Split into chunks if too long
            for i in range(0, len(token_ids), max_length):
                chunk = token_ids[i:i + max_length]
                if len(chunk) > 1:  # Need at least 2 tokens for input/target
                    self.tokens.append(chunk)
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            pad_id = self.tokenizer.pad_token_id
            tokens = tokens + [pad_id] * (self.max_length - len(tokens))
        
        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Input is all tokens except last, target is all tokens except first
        return tokens[:-1], tokens[1:]


def load_sample_data() -> List[str]:
    """Load sample training data"""
    # This is a placeholder - in practice, you'd load from files
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language.",
        "Machine learning models require lots of data.",
        "Transformers have revolutionized natural language processing.",
        "Deep learning is a subset of machine learning.",
        "Artificial intelligence will shape the future.",
        "Neural networks are inspired by the human brain.",
        "Large language models can generate human-like text.",
        "Training data quality is crucial for model performance.",
        "Attention mechanisms help models focus on relevant information."
    ] * 100  # Repeat to have more training data
    
    return sample_texts


def train_epoch(model: NanoDeepSeek, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, loss = model(inputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def validate(model: NanoDeepSeek, dataloader: DataLoader, device: torch.device) -> float:
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    logger.info("Starting validation")

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, loss = model(inputs, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train NanoDeepSeek model')
    parser.add_argument('--vocab_size', type=int, default=8192, help='Vocabulary size')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--data_file', type=str, help='Path to training data file')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load and prepare data
    if args.data_file and os.path.exists(args.data_file):
        with open(args.data_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        logger.info("No data file provided, using sample data")
        texts = load_sample_data()
    
    logger.info(f"Loaded {len(texts)} text samples")
    
    # Initialize pre-trained tokenizer (GPT-2)
    tokenizer = build_tokenizer("gpt2")
    actual_vocab_size = tokenizer.vocab_size
    logger.info(f"Using GPT-2 tokenizer with vocabulary size: {actual_vocab_size}")
    
    # Create datasets
    train_size = int(0.9 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    train_dataset = TextDataset(train_texts, tokenizer, args.max_seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, args.max_seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = NanoDeepSeek(
        vocab_size=actual_vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    logger.info(f"Model initialized with {model.get_num_params():,} parameters")
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_dataloader, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'vocab_size': actual_vocab_size,
                    'dim': args.dim,
                    'n_layers': args.n_layers,
                    'n_heads': args.n_heads,
                    'max_seq_len': args.max_seq_len
                }
            }
            
            torch.save(checkpoint, save_dir / "best_model.pt")
            logger.info(f"Saved best model checkpoint (val_loss: {val_loss:.4f})")
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': {
                'vocab_size': actual_vocab_size,
                'dim': args.dim,
                'n_layers': args.n_layers,
                'n_heads': args.n_heads,
                'max_seq_len': args.max_seq_len
            }
        }
        
        torch.save(checkpoint, save_dir / "latest_model.pt")
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
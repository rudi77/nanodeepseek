"""
Text generation script for NanoDeepSeek model.
"""

import torch
import argparse
from pathlib import Path
import logging

from nanodeepseek.model.model import NanoDeepSeek
from nanodeepseek.data.tokenizer import SimpleTokenizer


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: torch.device):
    """Load trained model and tokenizer"""
    
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    logger.info(f"Loaded tokenizer with vocab size: {tokenizer.get_vocab_size()}")
    
    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize model
    model = NanoDeepSeek(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model with {model.get_num_params():,} parameters")
    logger.info(f"Model trained for {checkpoint['epoch'] + 1} epochs")
    logger.info(f"Final validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model, tokenizer


def generate_text(
    model: NanoDeepSeek,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: torch.device = None
) -> str:
    """Generate text from a prompt"""
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Prompt tokens: {len(prompt_tokens)}")
    
    # Generate
    with torch.no_grad():
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode generated text
    generated_token_ids = generated_tokens[0].tolist()
    generated_text = tokenizer.decode(generated_token_ids)
    
    return generated_text


def interactive_mode(model: NanoDeepSeek, tokenizer: SimpleTokenizer, device: torch.device):
    """Interactive text generation mode"""
    
    print("🚀 NanoDeepSeek Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 50)
    
    temperature = 1.0
    top_k = None
    top_p = None
    max_tokens = 100
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            elif user_input.lower() == 'help':
                print("Commands:")
                print("  temp <value>    - Set temperature (default: 1.0)")
                print("  topk <value>    - Set top-k (default: None)")
                print("  topp <value>    - Set top-p (default: None)")
                print("  tokens <value>  - Set max tokens (default: 100)")
                print("  help           - Show this help")
                print("  quit           - Exit")
                print("\nCurrent settings:")
                print(f"  Temperature: {temperature}")
                print(f"  Top-k: {top_k}")
                print(f"  Top-p: {top_p}")
                print(f"  Max tokens: {max_tokens}")
                continue
            
            elif user_input.startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except (ValueError, IndexError):
                    print("Invalid temperature value")
                continue
            
            elif user_input.startswith('topk '):
                try:
                    top_k = int(user_input.split()[1])
                    print(f"Top-k set to {top_k}")
                except (ValueError, IndexError):
                    print("Invalid top-k value")
                continue
            
            elif user_input.startswith('topp '):
                try:
                    top_p = float(user_input.split()[1])
                    print(f"Top-p set to {top_p}")
                except (ValueError, IndexError):
                    print("Invalid top-p value")
                continue
            
            elif user_input.startswith('tokens '):
                try:
                    max_tokens = int(user_input.split()[1])
                    print(f"Max tokens set to {max_tokens}")
                except (ValueError, IndexError):
                    print("Invalid max tokens value")
                continue
            
            # Generate text
            if user_input:
                print("Generating...")
                generated = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user_input,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    device=device
                )
                print(f"\n📝 Generated:")
                print(generated)
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate text with NanoDeepSeek model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer file')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, help='Top-p (nucleus) sampling')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if files exist
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return
    
    if not Path(args.tokenizer).exists():
        logger.error(f"Tokenizer file not found: {args.tokenizer}")
        return
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, device)
    else:
        # Single generation
        if not args.prompt:
            logger.error("Please provide a prompt with --prompt or use --interactive mode")
            return
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        
        print("Generated text:")
        print("-" * 50)
        print(generated)


if __name__ == "__main__":
    main()
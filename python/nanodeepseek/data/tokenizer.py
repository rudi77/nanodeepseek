from  transformers import AutoTokenizer

def build_tokenizer(name: str = "gpt2") -> AutoTokenizer:
    """
    Build and return a tokenizer based on the specified model name.
    
    Args:
        model_name (str): The name of the pre-trained model for which to load the tokenizer.
        
    Returns:
        AutoTokenizer: The tokenizer corresponding to the specified model.
    """
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    
    # GPT-2 has no pad token by default, so we set it to eos token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


if __name__ == "__main__":
    # Example usage
    tokenizer = build_tokenizer("gpt2")
    sample_text = "Hello, NanoDeepSeek!"
    tokens = tokenizer.encode(sample_text, add_special_tokens=True)
    print("Sample text:", sample_text)
    print("Token IDs:", tokens)

    # decode back to text
    decoded_text = tokenizer.decode(tokens)
    print("Decoded text:", decoded_text)
    pass
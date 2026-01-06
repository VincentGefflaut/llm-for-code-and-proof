from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- EXERCISE 1: La disparition (No 'e' or 'E) ---
class LaDisparition:
    """
    Generate text without ever using the letter 'e' or 'E'.
    For this, you must use model() directly: model(input_ids) yields logits.
    You need to manually adjust the logits to forbid tokens containing 'e' or 'E'.
    REQUIREMENT: Do NOT use model.generate().
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Here you want to pre-calculate forbidden token IDs

        # Warning: The evaluation server uses a different model and tokenizer than the template. Do not hard-code Token IDs. Use self.tokenizer.get_vocab() or self.tokenizer.encode() to find the IDs relevant to the current model.

    def __call__(self, prompt, max_tokens=30):
        # Tokenize input prompt:
        # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Generate tokens manually, one step at a time:
        # (The bulk of the logic goes here)
        # Hint: generating a single answer may not be enough!
    
        # Decode output tokens to string and return                   
        # return tokenizer.decode(generated, skip_special_tokens=True)
        pass


# --- EXERCISE 2: The Toulouse Sequence ---
class ToulouseSequence:
    """
    Generate text without ever using the word 'Toulouse'.
    For this, you must use model() directly: model(input_ids) yields logits.
    You need to manually adjust the logits. It is more difficult here because
    'Toulouse' is a multi-token word.
    REQUIREMENT: Do NOT use model.generate().
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Here you want to pre-calculate forbidden token IDs
        # Hint:
        # print(tokenizer.encode("Toulouse", add_special_tokens=False))

    def __call__(self, prompt, max_tokens=30):
        # Tokenize input prompt:
        # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Generate tokens manually, one step at a time:
        # (The bulk of the logic goes here)
        # Hint: you need to track partial matches of the forbidden word
    
        # Decode output tokens to string and return                   
        # return tokenizer.decode(generated, skip_special_tokens=True)
        pass
    
if __name__ == "__main__":
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="auto")
    la_disparition_generator = LaDisparition(model, tokenizer)
    print("Ex 1 (No 'e'):", la_disparition_generator("Describe a cat."))
    toulouse_sequence_generator = ToulouseSequence(model, tokenizer)
    print("Ex 2 (No 'Toulouse'):", toulouse_sequence_generator("The pink city in France is"))

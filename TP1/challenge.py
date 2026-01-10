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
        self.forbidden_ids = []
        self.start_with_letter_ids = []
        self.end_with_letter_ids = []
        for word,id in tokenizer.get_vocab().items():
            if 'e' in word or 'E' in word:
                self.forbidden_ids.append(id)
                # print(word,id)
            if word[0].isalpha():
                self.start_with_letter_ids.append(id)
            if word[-1].isalpha():
                self.end_with_letter_ids.append(id)

        # Warning: The evaluation server uses a different model and tokenizer than the template. Do not hard-code Token IDs. Use self.tokenizer.get_vocab() or self.tokenizer.encode() to find the IDs relevant to the current model.

    def __call__(self, prompt, max_tokens=30):
        # Tokenize input prompt:
        # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Generate tokens manually, one step at a time:
        # (The bulk of the logic goes here)
        # Hint: generating a single answer may not be enough!
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device).view(-1)
        word_start_idx = None # The current word's first token idx in input_ids
        word_start_alternatives = None # Alternative tokens to word start
        prompt_length = input_ids.shape[0]

        while input_ids.shape[0] < max_tokens + prompt_length:
            generated = self.model(input_ids.view(1,-1))
            next_token_id = torch.argmax(generated.logits[0][-1])
            # Check if a new word has started ie there is no continuation
            if not (next_token_id in self.start_with_letter_ids and input_ids[-1] in self.end_with_letter_ids):
                word_start_idx = input_ids.shape[0]
                word_start_alternatives = torch.argsort(generated.logits[0][-1], descending=True)
            if next_token_id in self.forbidden_ids and word_start_idx is not None:
                input_ids = input_ids[:word_start_idx]
                i=1
                while word_start_alternatives[i] in self.forbidden_ids:
                    i += 1
                word_start_alternatives = word_start_alternatives[i:]
                next_token_id = word_start_alternatives[0]
            input_ids = torch.cat((input_ids, next_token_id.view(-1)))

        generated = input_ids
    
        # Decode output tokens to string and return                   
        return self.tokenizer.decode(generated, skip_special_tokens=True)


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

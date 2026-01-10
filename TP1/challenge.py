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
        self.forbidden_ids_mask = torch.zeros(len(tokenizer.get_vocab()), dtype=torch.bool, device=model.device)
        self.start_with_letter_mask = torch.zeros(len(tokenizer.get_vocab()), dtype=torch.bool, device=model.device)
        self.end_with_letter_mask = torch.zeros(len(tokenizer.get_vocab()), dtype=torch.bool, device=model.device)
        for word,id in tokenizer.get_vocab().items():
            if 'e' in word or 'E' in word:
                self.forbidden_ids_mask[id] = True
            if word[0].isalpha():
                self.start_with_letter_mask[id] = True
            if word[-1].isalpha():
                self.end_with_letter_mask[id] = True

        # Warning: The evaluation server uses a different model and tokenizer than the template. Do not hard-code Token IDs. Use self.tokenizer.get_vocab() or self.tokenizer.encode() to find the IDs relevant to the current model.

    @torch.no_grad()
    def __call__(self, prompt, max_tokens=30):
        device = self.model.device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device).view(-1)
        word_start_state = {
            'input_ids': None,
            'past_key_values': None,
            'alternatives': None
        }
        prompt_length = input_ids.shape[0]
        past_key_values = None
        
        while input_ids.shape[0] < max_tokens + prompt_length:
            if past_key_values is None:
                model_input = input_ids.view(1,-1)
            else:
                model_input = input_ids[-1:].view(1,-1)
            
            generated = self.model(model_input, past_key_values=past_key_values, use_cache=True)

            logits = generated.logits[0][-1]
            logits.masked_fill_(self.forbidden_ids_mask, float('-inf'))
            
            next_token_id = torch.argmax(logits).item()

            # Check if a new word has started ie there is no continuation
            if not (self.start_with_letter_mask[next_token_id] and self.end_with_letter_mask[input_ids[-1]]):
                word_start_state['input_ids'] = input_ids.clone()
                word_start_state['past_key_values'] = past_key_values # Cache avant le mot
                word_start_state['alternatives'] = torch.argsort(logits).tolist()
            
            # If the next token is forbidden, backtrack to word start and try next alternative
            if self.forbidden_ids_mask[next_token_id] and word_start_state['input_ids'] is not None:
                input_ids = word_start_state['input_ids']
                past_key_values = word_start_state['past_key_values']
                # Try next alternative (pop the highest probability one)
                next_token_id = word_start_state['alternatives'].pop()
            else:
                input_ids = torch.cat((input_ids, torch.tensor([next_token_id], device=device)))
                past_key_values = generated.past_key_values

        return self.tokenizer.decode(input_ids, skip_special_tokens=True)


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
        self.forbidden_sequence = self.tokenizer.encode("Toulouse", add_special_tokens=False)

    @torch.no_grad()
    def __call__(self, prompt, max_tokens=30):
        device = self.model.device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device).view(-1)
        suspicious_start_state = {
            'input_ids': None,
            'past_key_values': None,
            'alternatives': None,
            'match_index': 0
        }
        prompt_length = input_ids.shape[0]
        past_key_values = None
        
        while input_ids.shape[0] < max_tokens + prompt_length:
            if past_key_values is None:
                model_input = input_ids.view(1,-1)
            else:
                model_input = input_ids[-1:].view(1,-1)
            
            generated = self.model(model_input, past_key_values=past_key_values, use_cache=True)

            logits = generated.logits[0][-1]
            
            next_token_id = torch.argmax(logits).item()

            # Check if we're suspicious
            if next_token_id == self.forbidden_sequence[suspicious_start_state['match_index']]:
                suspicious_start_state['match_index'] += 1
                if suspicious_start_state['match_index'] == 0:
                    # Start of a new suspicious sequence
                    suspicious_start_state['input_ids'] = input_ids.clone()
                    suspicious_start_state['past_key_values'] = past_key_values # Cache avant le mot
                    suspicious_start_state['alternatives'] = torch.argsort(logits).tolist()
                elif suspicious_start_state['match_index'] == len(self.forbidden_sequence):
                    # Full forbidden sequence matched
                    suspicious_start_state['match_index'] = 0
                    input_ids = suspicious_start_state['input_ids']
                    past_key_values = suspicious_start_state['past_key_values']
                    # Try next alternative (pop the highest probability one)
                    next_token_id = suspicious_start_state['alternatives'].pop()
            else:
                input_ids = torch.cat((input_ids, torch.tensor([next_token_id], device=device)))
                past_key_values = generated.past_key_values

        return self.tokenizer.decode(input_ids, skip_special_tokens=True)
    
if __name__ == "__main__":
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="auto")
    la_disparition_generator = LaDisparition(model, tokenizer)
    print("Ex 1 (No 'e'):", la_disparition_generator("Describe a cat."))
    toulouse_sequence_generator = ToulouseSequence(model, tokenizer)
    print("Ex 2 (No 'Toulouse'):", toulouse_sequence_generator("The pink city in France is"))

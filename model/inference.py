import torch
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, MT5ForConditionalGeneration

# Isolates the inference function of a tuned model

class Inference:
    def __init__(self, hparams, multilingual: bool):
        self.hparams = argparse.Namespace(**hparams)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if multilingual:
            self.model = MT5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path).to(self.DEVICE)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path).to(self.DEVICE)

    def predict(self, sentence):
        tokenized_sentence = self.tokenizer(sentence, return_tensors="pt").to(self.DEVICE)
        input_ids = tokenized_sentence["input_ids"].to(self.DEVICE)
        attention_mask = tokenized_sentence["attention_mask"].to(self.DEVICE)
        generated = self.model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        ).squeeze()
        predictions = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return predictions





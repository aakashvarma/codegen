import logging
import torch
from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        )
from peft import LoraConfig


class FineTuning:
    def __init__(self):
        pass

    def get_model(self, model_config, base_model):
        model = AutoModelForCausalLM.from_pretrained(
                base_model, 
                quantization_config=model_config, 
                trust_remote_code=True
                )
        model.config.use_cache = False

        return model

    def get_tokenizer(self, base_model):
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        return tokenizer

    def fine_tuning_setup(self, base_model):
        pass

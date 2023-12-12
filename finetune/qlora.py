import logging
from finetune.finetune import FineTuning

import torch
from transformers import BitsAndBytesConfig


class QLoRA(FineTuning):
    def __init__(self):
        super().__init__()

    def fine_tuning_setup(self, base_model):
        logging.info("Setting up QLoRA configuration.")
        try:
            model_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = self.get_model(model_config, base_model)
            tokenizer = self.get_tokenizer(base_model)
                
            logging.info("QLoRA configuration successful.")
            
        except Exception as e:
            logging.error(f"Error in setting up QLoRA configuration: ", e)

        return model, tokenizer

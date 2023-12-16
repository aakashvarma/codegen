import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


class FineTuner:
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config

    def get_model(self):
        device_map = "auto"
        compute_dtype = (
            torch.float16
            if self.training_config.fp16
            else (torch.bfloat16 if self.training_config.bf16 else torch.float32)
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            cache_dir=self.model_config.cache_dir,
            load_in_4bit=self.model_config.bits == 4,
            load_in_8bit=self.model_config.bits == 8,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=self.model_config.bits == 4,
                load_in_8bit=self.model_config.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.model_config.double_quant,
                bnb_4bit_quant_type=self.model_config.quant_type,
            ),
            torch_dtype=(
                torch.float32
                if self.training_config.fp16
                else (torch.bfloat16 if self.training_config.bf16 else torch.float32)
            ),
            trust_remote_code=self.model_config.trust_remote_code,
            use_auth_token=self.model_config.use_auth_token,
        )
        model.config.use_cache = False

        return model

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        return tokenizer

    def finetuning_model_setup(self):
        """
        Set up Finetune configuration.

        Args:
            model_conifg (str): The model configuraton.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        logging.info("Setting up Finetuning configuration.")
        try:
            model = self.get_model()
            tokenizer = self.get_tokenizer()

            logging.info("Finetuning configuration successful.")

        except Exception as e:
            error_message = f"Error in setting up Finetuning configuration: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message)

        return model, tokenizer

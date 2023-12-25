import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

class Quantizer:
    """
    Class for setting up and configuring a model for fine-tuning.

    Args:
        model_config: Model configuration object.

    Methods:
        get_model(): Get the fine-tuning model based on the provided configuration.
        get_tokenizer(): Get the tokenizer for the fine-tuning model.
        model_setup(): Set up fine-tuning configuration.

    Attributes:
        model_config: Model configuration object.
    """
    def __init__(self, model_config):
        self.model_config = model_config

    def get_model(self):
        """
        Get the fine-tuning model based on the provided configuration.

        Returns:
            transformers.AutoModelForCausalLM: The fine-tuning model.
        """
        device_map = "auto"
        compute_dtype = (
            torch.float16
            if self.model_config.compute_type == "fp16"
            else (
                torch.bfloat16
                if self.model_config.compute_type == "bf16"
                else torch.float32
            )
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
                if self.model_config.compute_type == "fp16"
                else (
                    torch.bfloat16
                    if self.model_config.compute_type == "bf16"
                    else torch.float32
                )
            ),
            trust_remote_code=self.model_config.trust_remote_code,
            use_auth_token=self.model_config.use_auth_token,
        )
        model.config.use_cache = False

        return model

    def get_tokenizer(self):
        """
        Get the tokenizer for the fine-tuning model.

        Returns:
            transformers.AutoTokenizer: The fine-tuning model tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        return tokenizer

    def model_setup(self):
        """
        Set up Finetune configuration.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        logging.info("Setting up Finetuning configuration.")
        try:
            model = self.get_model()
            tokenizer = self.get_tokenizer()

            logging.info("Finetuning configuration successful.")

        except Exception as e:
            logging.error(f"Error in setting up Finetuning configuration: {e}")
            raise

        return model, tokenizer

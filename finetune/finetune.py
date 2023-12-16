import logging
import torch
from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        )
from peft import LoraConfig


class FineTuning:
    def __init__(self):
        pass

    def get_model(self, model_config, base_model):
        # model = AutoModelForCausalLM.from_pretrained(
        #         base_model, 
        #         quantization_config=model_config, 
        #         trust_remote_code=True,
        #         device_map = "auto",
        #         )
        model =  AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=model_config.cache_dir,
            load_in_4bit=model_config.bits == 4,
            load_in_8bit=model_config.bits == 8,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=model_config.bits == 4,
                load_in_8bit=model_config.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=model_config.double_quant,
                bnb_4bit_quant_type=model_config.quant_type,
            ),
            torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
            trust_remote_code=model_config.trust_remote_code,
            use_auth_token=model_config.use_auth_token
        )
        model.config.use_cache = False

        return model

    def get_tokenizer(self, base_model):
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        return tokenizer

    def finetuning_model_setup(self, base_model):
        """
        Set up Finetune configuration.

        Args:
            base_model (str): The base model ID.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
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
            error_message = f"Error in setting up QLoRA configuration: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message)

        return model, tokenizer

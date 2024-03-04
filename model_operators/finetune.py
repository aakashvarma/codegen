import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from model_operators.quantize import Quantizer


class FineTuner(Quantizer):
    def __init__(self, model_config, finetune_config):
        self.model_config = model_config
        self.finetune_config = finetune_config

    def prepare_model(self, model):
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        lora_config = LoraConfig(
            r=self.finetune_config.r,
            lora_alpha=self.finetune_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.finetune_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        return model

    def model_setup(self, model_path, model_with_adapter, merge_model, llm_int8):
        logging.info("Setting up Finetuning configuration.")
        try:
            model = self.get_model(model_path, model_with_adapter, merge_model, llm_int8)
            model = self.prepare_model(model)
            tokenizer = self.get_tokenizer()

            logging.info("Finetuning configuration successful.")

        except Exception as e:
            logging.error(f"Error in setting up Finetuning configuration: {e}")
            raise

        return model, tokenizer

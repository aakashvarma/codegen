import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from model_operators.quantize import Quantizer


class FineTuner(Quantizer):
    """
    A class for fine-tuning models with quantization and additional modifications using the PEFT framework.

    Attributes:
        model_config (object): The configuration object for the base model.
        finetune_config (object): The configuration object for the fine-tuning process.

    Methods:
        __init__(self, model_config, finetune_config):
            Initializes the FineTuner with the provided model and fine-tuning configurations.

        prepare_model(self, model):
            Prepares the given model for fine-tuning by applying the PEFT framework with specified configurations.

        model_setup(self):
            Sets up the fine-tuning configuration, including loading the model and tokenizer.
    """

    def __init__(self, model_config, finetune_config):
        """
        Initializes the FineTuner.

        Args:
            model_config (object): The configuration object for the base model.
            finetune_config (object): The configuration object for the fine-tuning process.
        """
        self.model_config = model_config
        self.finetune_config = finetune_config

    def prepare_model(self, model):
        """
        Prepares the given model for fine-tuning by applying the PEFT framework with specified configurations.

        Args:
            model: The base model to be fine-tuned.

        Returns:
            model: The prepared model for fine-tuning.
        """
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

    def model_setup(self):
        """
        Sets up the fine-tuning configuration, including loading the model and tokenizer.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        logging.info("Setting up Finetuning configuration.")
        try:
            model = self.get_model()
            model = self.prepare_model(model)
            tokenizer = self.get_tokenizer()

            logging.info("Finetuning configuration successful.")

        except Exception as e:
            logging.error(f"Error in setting up Finetuning configuration: {e}")
            raise

        return model, tokenizer

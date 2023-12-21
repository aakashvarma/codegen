import logging
import sys
import torch

sys.path.append("../utils")
sys.path.append("../model_operators")
sys.path.append("../trainer")

from model_operators.finetune import Quantizer, FineTuner
from trainer.trainer import LLMTrainer


class Model:
    """
    Class for working with a pretrained language model and performing inference.

    Args:
        model_config: Model configuration object.

    Methods:
        __init__(model_config): Initialize the Model class with attributes.
        __str__(): Provide a string representation of the Model instance.
        get_model_and_tokenizer(): Download a pretrained model and tokenizer based on the given base model ID using the selected adapter.
        infer_model(model, tokenizer, prompt): Perform model inference on the provided prompt.

    Attributes:
        model_config: Model configuration object.
    """

    def __init__(self, model_config, trainer_config=None, finetune_config=None):
        """
        Initialize the Model class with attributes.

        Args:
            model_config: Model configuration object.
        """
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.finetune_config = finetune_config
        self.model = None
        self.tokenizer = None

    def __str__(self):
        """
        Provide a string representation of the Model instance.

        Returns:
            str: String representation of the Model instance.
        """
        return f"Model Config: {self.model_config}"

    def get_inference_model_and_tokenizer(self):
        """
        Download a pretrained model and tokenizer based on the given base model ID using the selected adapter.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        try:
            logging.info("Setting up model.")
            quantizer = Quantizer(self.model_config)

            self.model, self.tokenizer = quantizer.model_setup()
            return self.model, self.tokenizer

        except Exception as e:
            error_message = f"Error on downloading the model: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message)
    
    def get_finetuning_model_and_tokenizer(self):
        """
        Download a pretrained model and tokenizer based on the given base model ID using the selected adapter.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        try:
            logging.info("Setting up model.")
            finetuner = FineTuner(self.model_config, self.finetune_config)

            self.model, self.tokenizer = finetuner.model_setup()
            return self.model, self.tokenizer

        except Exception as e:
            error_message = f"Error on downloading the model: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message)

    def infer_model(self, prompt):
        """
        Perform model inference on the provided prompt.

        Args:
            model: The pretrained language model.
            tokenizer: The tokenizer associated with the model.
            prompt (str): The input prompt for inference.

        Returns:
            str: Inference output.
        """
        try:
            self.get_inference_model_and_tokenizer()
            if prompt:
                logging.info("Running model inference on the prompt.")
                model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")

                self.model.eval()
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **model_input, max_new_tokens=100
                    )[0]
                    decoded_output = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    logging.info("Inference Output: %s", decoded_output)
                    return decoded_output
            else:
                error_message = "Prompt cannot be empty."
                logging.error(error_message)
                raise ValueError(error_message)

        except Exception as e:
            error_message = f"Error during model inference: {e}"
            logging.error(error_message, exc_info=True)
            raise RuntimeError(error_message)
        
    def finetune_model(self):
        """
        Fine-tune the model with the given training and validation data.

        Parameters:
            train_data (Dataset): Training data.
            val_data (Optional[Dataset]): Validation data.
            trainer_config (TrainingArguments): Training configuration.
        """
        try:
            self.get_finetuning_model_and_tokenizer()
            trainer_obj = LLMTrainer(self.model, self.tokenizer, self.trainer_config)
            trainer = trainer_obj.get_trianer()
            trainer.train()
            self.model.save_pretrained(self.trainer_config.model_output_dir)

            logging.info("Finetuned model saved to: %s", self.trainer_config.model_output_dir)
        except Exception as e:
            error_message = f"Error in model training: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message)


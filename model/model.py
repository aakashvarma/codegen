import logging
import sys
import torch
from model_operators.finetune import Quantizer, FineTuner
from trainer.trainer import LLMTrainer

sys.path.append("../utils")
sys.path.append("../model_operators")
sys.path.append("../trainer")


class Model:
    """
    Class for working with a pretrained language model and performing inference.

    Args:
        model_config: Model configuration object.
        trainer_config: Trainer configuration object.
        finetune_config: Fine-tune configuration object.

    Methods:
        __init__(model_config, trainer_config=None, finetune_config=None): Initialize the Model class with attributes.
        __str__(): Provide a string representation of the Model instance.
        get_inference_model_and_tokenizer(): Download a pretrained model and tokenizer based on the given base model ID using the selected adapter for inference.
        get_finetuning_model_and_tokenizer(): Download a pretrained model and tokenizer based on the given base model ID using the selected adapter for fine-tuning.
        infer_model(prompt): Perform model inference on the provided prompt.
        finetune_model(): Fine-tune the model with the given training and validation data.

    Attributes:
        model_config: Model configuration object.
        trainer_config: Trainer configuration object.
        finetune_config: Fine-tune configuration object.
        model: Pretrained language model.
        tokenizer: Tokenizer associated with the model.
    """

    def __init__(self, model_config, trainer_config=None, finetune_config=None):
        """
        Initialize the Model class with attributes.

        Args:
            model_config: Model configuration object.
            trainer_config: Trainer configuration object.
            finetune_config: Fine-tune configuration object.
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
        Download a pretrained model and tokenizer based on the given base model ID using the selected adapter for inference.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        try:
            logging.info("Setting up model for inference.")
            quantizer = Quantizer(self.model_config)
            self.model, self.tokenizer = quantizer.model_setup()
            return self.model, self.tokenizer

        except Exception as e:
            error_message = f"Error on setting up the model for inference: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message) from e

    def get_finetuning_model_and_tokenizer(self):
        """
        Download a pretrained model and tokenizer based on the given base model ID using the selected adapter for fine-tuning.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        try:
            logging.info("Setting up model for fine-tuning.")
            finetuner = FineTuner(self.model_config, self.finetune_config)
            self.model, self.tokenizer = finetuner.model_setup()
            return self.model, self.tokenizer

        except Exception as e:
            error_message = f"Error on setting up the model for fine-tuning: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message) from e

    def infer_model(self, prompt):
        """
        Perform model inference on the provided prompt.

        Args:
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
                    logging.info("Model inference done.")
                    return decoded_output
            else:
                error_message = "Prompt cannot be empty."
                logging.error(error_message)
                raise ValueError(error_message)

        except Exception as e:
            error_message = f"Error during model inference: {e}"
            logging.error(error_message, exc_info=True)
            raise RuntimeError(error_message) from e

    def finetune_model(self):
        """
        Fine-tune the model with the given training and validation data.

        Raises:
            RuntimeError: If an error occurs during fine-tuning.
        """
        try:
            self.get_finetuning_model_and_tokenizer()
            trainer_obj = LLMTrainer(
                self.model, self.tokenizer, self.trainer_config
            )
            trainer = trainer_obj.get_trainer()
            trainer.train()
            self.model.save_pretrained(self.trainer_config.model_output_dir)

            logging.info(
                "Finetuned model saved to: %s", self.trainer_config.model_output_dir
            )
        except Exception as e:
            error_message = f"Error in model fine-tuning: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message) from e

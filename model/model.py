import logging
import sys
import torch

sys.path.append("../utils")
sys.path.append("../finetune")

from finetune.finetune import FineTuner

class Model:
    def __init__(self, model_config):
        """
        Initialize the Model class with attributes.
        """
        self.model_config = model_config

    def __str__(self):
        """
        Provide a string representation of the Model instance.
        """
        return (
            f"Model Config: {self.model_config}"
        )

    def get_model_and_tokenizer(self):
        """
        Download a pretrained model and tokenizer based on the given base model ID using the selected adapter.
        """
        try:
            logging.info("Setting up model.")
            finetuner = FineTuner(self.model_config)

            model, tokenizer = finetuner.finetuning_model_setup()
            return model, tokenizer

        except Exception as e:
            error_message = f"Error on downloading the model: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message)

    def model_inference(self, model, tokenizer, prompt):
        """
        Perform model inference on the provided prompt.
        """
        try:
            if prompt:
                logging.info("Running model inference on the prompt.")
                model_input = tokenizer(prompt, return_tensors="pt").to("cuda")

                model.eval()
                with torch.no_grad():
                    generated_tokens = model.generate(**model_input, max_new_tokens=100)[0]
                    decoded_output = tokenizer.decode(
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
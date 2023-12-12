import logging
import sys
import torch
import argparse

sys.path.append("utils")
sys.path.append("finetune")

from utils import parse_config_json, parse_prompt_text
from qlora import QLoRA
from lora import LoRA


class Model:
    def __init__(self):
        """
        Initialize the Model class with attributes.
        """
        self.model_name = None
        self.adapter = None
        self.prompt = None
        self.model = None
        self.tokenizer = None

    def __str__(self):
        """
        Provide a string representation of the Model instance.
        """
        return f"Model Name: {self.model_name}"

    def parse_args(self):
        """
        Parse model configuration dictionary to set model name and choose the appropriate adapter based on the configuration.
        """
        logging.info("Parsing arguments.")
        parser = argparse.ArgumentParser(description='Run model inference with prompt.')
        parser.add_argument('json_file_path', type=str, help='Path to the JSON file containing model configuration.')
        parser.add_argument('prompt_file_path', type=str, help='Path to the file containing the prompt text.')

        args = parser.parse_args()

        model_config_dict = parse_config_json(args.json_file_path)
        self.prompt = parse_prompt_text(args.prompt_file_path)
    
        self.model_name = model_config_dict.get("model_name")
        config_mapping = {
            "lora_config": LoRA, 
            "qlora_config": QLoRA
        }
        try:
            self.adapter = config_mapping[model_config_dict.get("adapter_type")]
            logging.info("Adapter setup successful.")
        except KeyError as e:
            error_message = f"Invalid adapter configuration. Either 'lora_config' or 'qlora_config' should be present: {e}"
            logging.error(error_message)
            raise ValueError(error_message)

    def setup_model(self):
        """
        Download a pretrained model and tokenizer based on the given base model ID using the selected adapter.
        """
        try:
            logging.info("Setting up model.")
            base_model = self.model_name
            adapter = self.adapter()
            self.model, self.tokenizer = adapter.fine_tuning_setup(base_model)
        except Exception as e:
            error_message = f"Error on downloading the model: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message)

    def model_inference(self):
        """
        Perform model inference on the provided prompt.
        """
        if self.prompt:
            logging.info("Running model inference on the prompt.")
            model_input = self.tokenizer(self.prompt, return_tensors="pt").to("cuda")

            self.model.eval()
            with torch.no_grad():
                generated_tokens = self.model.generate(
                        **model_input, max_new_tokens=100
                        )[0]
                decoded_output = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                        )
                print(decoded_output)
        else:
            error_message = "Prompt cannot be empty."
            logging.error(error_message)
            raise ValueError(error_message)

    def runner(self):
        """
        Main entry point for the script. Parse command line arguments, load model configuration, and perform inference.
        """
        self.parse_args()
        self.setup_model()
        self.model_inference()


if __name__ == "__main__":
    model_instance = Model()
    try:
        model_instance.runner()
    except ValueError as ve:
        print(f"Error: {ve}")
    except RuntimeError as re:
        print(f"Runtime Error: {re}")

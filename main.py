import logging
import sys
import torch

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

    def parse_args(self, model_config_dict):
        """
        Parse model configuration dictionary to set model name and choose the appropriate adapter based on the configuration.
        """
        self.model_name = model_config_dict.get("model_name")
        config_mapping = {"lora_config": LoRA, "qlora_config": QLoRA}
        for config_key, adapter_class in config_mapping.items():
            if config_key in model_config_dict.get("adapter_type", []):
                self.adapter = adapter_class()
                logging.info("Adapter setup successful.")
                break
        else:
            message = "Invalid configuration. Either 'lora_config' or 'qlora_config' should be present."
            logging.error(message)

    def setup_model(self):
        """
        Download a pretrained model and tokenizer based on the given base model ID using the selected adapter.
        """
        try:
            base_model = self.model_name
            self.model, self.tokenizer = self.adapter.fine_tuning_setup(base_model)
        except Exception as e:
            logging.error(f"Error on downloading the model: {e}")

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
            logging.error("Prompt cannot be empty.")

    def runner(self):
        """
        Main entry point for the script. Parse command line arguments, load model configuration, and perform inference.
        """
        if len(sys.argv) != 3:
            print("Usage: python main.py <json_file_path> <prompt_file_path>")
            sys.exit(1)

        model_config_dict = parse_config_json(sys.argv[1])
        self.prompt = parse_prompt_text(sys.argv[2])

        self.parse_args(model_config_dict)
        self.setup_model()
        self.model_inference()


if __name__ == "__main__":
    model_instance = Model()
    model_instance.runner()

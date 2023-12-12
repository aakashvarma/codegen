import logging
import sys
import torch

sys.path.append("utils")
sys.path.append("finetune")

from utils import parse_config_json, parse_prompt_text
from qlora import QLoRA
from lora import LoRA


class Model:
    def __init__(self, ):
        self.model_name = None
        self.adapter = None
        self.prompt = None
        self.model = None
        self.tokenizer = None

    def __str__(self):
        return f"model_name: {self.model_name}"
    
    def parse_args(self, model_config_instance):
        self.model_name = model_config_instance["model_name"]
        config_mapping = {
            'lora_config': LoRA,
            'qlora_config': QLoRA,
        }
        for config_key, adapter_class in config_mapping.items():
            if config_key in model_config_instance["adapter_type"]:
                self.adapter = adapter_class()
                logging.info("Adapter setup successful.")
                break
        else:
            message = f"Invalid configuration. Either 'lora_config' or 'qlora_config' should be present."
            logging.error(message)

    def setup_model(self):
        """
        Downloads a pretrained model and tokeniser based on the given base model ID.
        """
        try:
            base_model = self.model_name
            model, tokenizer = self.adapter.fine_tuning_setup()
        except Exception as e:
            logging.error(f"Error on downloading the model: ", e)
                 
    def model_inference(self):
        if self.prompt:
            logging.info("Running model inference on the prompt.")
            model_input = self.tokenizer(self.prompt, return_tensors="pt").to("cuda")
          
            self.model.eval()
            with torch.no_grad():
                print(
                    self.tokenizer.decode(
                        self.model.generate(**model_input, max_new_tokens=100)[0],
                        skip_special_tokens=True,
                    )
                )
            
        else:
            logging.error("Prompt cannot be empty.")
    
    def runner(self):
        if len(sys.argv) != 3:
            print("Usage: python main.py <json_file_path> <prompt_file_path>")
            sys.exit(1)

        model_config_instance = parse_config_json(sys.argv[1])
        self.prompt = parse_prompt_text(sys.argv[2])
        

import logging
import sys
import argparse
import yaml
from typing import Optional, Union


sys.path.append("model")
sys.path.append("utils")
sys.path.append("finetune")

from model import Model
from utils import parse_prompt_text

logging.basicConfig(level=logging.DEBUG)

class ModelArguments:
    def __init__(
        self,
        model_name: Optional[str] = "meta-llama/Llama-2-7b-hf",
        cache_dir: Optional[str] = None,
        r: Optional[int] = 64,
        lora_alpha: Optional[float] = 32,
        lora_dropout: Optional[float] = 0.05,
        bits: Optional[int] = 4,
        double_quant: Optional[bool] = True,
        quant_type: str = "nf4",
        trust_remote_code: Optional[bool] = False,
        use_auth_token: Union[bool, str] = False,
        compute_type: Optional[str] = "bf16",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bits = bits
        self.double_quant = double_quant
        self.quant_type = quant_type
        self.trust_remote_code = trust_remote_code
        self.use_auth_token = use_auth_token
        self.compute_type = compute_type

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))
    
    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["model_arguments"]
        return cls(**yaml_args)


class DataTrainingArguments:
    def __init__(
        self,
        dataset_name: Optional[str] = "Dahoas/full-hh-rlhf",
        block_size: Optional[int] = 4096,
        multi_gpu: Optional[bool] = False,
        tensor_parallel: Optional[bool] = False,
        model_output_dir: Optional[str] = "LLaMA/LoRA",
    ):
        self.dataset_name = dataset_name
        self.block_size = block_size
        self.multi_gpu = multi_gpu
        self.tensor_parallel = tensor_parallel
        self.model_output_dir = model_output_dir

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))
    
    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["data_training_arguments"]
        return cls(**yaml_args)


class Runner:
    def __init__(self) -> None:
        pass

    def get_parser(self):
        parser = argparse.ArgumentParser(description="Arguments")
        parser.add_argument("--yaml_path", required=True, help="Path to the YAML file containing both sets of arguments.")
        parser.add_argument("--prompt_file", required=True, help="Path to the text file containing the prompt.")
        parser.add_argument("--infer", action="store_true", help="Perform inference.")
        parser.add_argument("--finetune", action="store_true", help="Perform finetuning.")
        return parser

    def infer(self, model_config, prompt):
        """
        Main entry point for the script. Parse command line arguments, load model configuration, and perform inference.
        """
        try:
            model_infer = Model(model_config)
            model, tokenizer = model_infer.get_model_and_tokenizer()
            output = model.infer.model_inference(model, tokenizer, prompt)
            logging.info("Inference Output: %s", output)
            return output
        except Exception as e:
            logging.error("Error during inference: %s", e, exc_info=True)
            raise e

    def finetune(self, model_config, data_training_config):
        """
        Placeholder for the finetuning function. Implement this function based on your finetuning logic.
        """
        logging.info("Finetuning function placeholder. Implement based on your logic.")

    def main(self):
        try:
            args = self.get_parser().parse_args()

            # Set up loggers
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            debug_logger = logging.getLogger("debug_logger")
            error_logger = logging.getLogger("error_logger")

            logger.info("Starting the script.")

            # Create instances of the classes from the YAML file
            model_arguments = ModelArguments.from_yaml(args.yaml_path)
            data_training_arguments = DataTrainingArguments.from_yaml(args.yaml_path)

            logger.info("Model Arguments:")
            logger.info(model_arguments.__dict__)

            logger.info("Data Training Arguments:")
            logger.info(data_training_arguments.__dict__)

            # Read the prompt from the text file
            with open(args.prompt_file, "r") as prompt_file:
                prompt = prompt_file.read()

            if args.infer:
                result = self.infer(model_arguments, prompt)
                logger.info("Script completed successfully with result: %s", result)
            elif args.finetune:
                self.finetune(model_arguments, data_training_arguments)
                logger.info("Script completed finetuning successfully.")

        except ValueError as ve:
            error_logger.error("ValueError: %s", ve)
        except RuntimeError as re:
            error_logger.error("RuntimeError: %s", re)
        except Exception as e:
            error_logger.error("An unexpected error occurred: %s", e, exc_info=True)


if __name__ == "__main__":
    Runner().main()
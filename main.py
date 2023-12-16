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
        return parser


    def infer(self, model):
        """
        Main entry point for the script. Parse command line arguments, load model configuration, and perform inference.
        """
        model.setup_model()


if __name__ == "__main__":
    try:
        runner = Runner()
        args = runner.get_parser().parse_args()

        # Create instances of the classes from the YAML file
        model_arguments = ModelArguments.from_yaml(args.yaml_path)
        data_training_arguments = DataTrainingArguments.from_yaml(args.yaml_path)

        # Set up loggers
        logger = logging.getLogger(__name__)
        debug_logger = logging.getLogger("debug_logger")
        error_logger = logging.getLogger("error_logger")

        logger.info("Model Arguments:")
        logger.info(model_arguments.__dict__)

        logger.info("Data Training Arguments:")
        logger.info(data_training_arguments.__dict__)

    except ValueError as ve:
        error_logger.error(f"Error: {ve}")
    except RuntimeError as re:
        error_logger.error(f"Runtime Error: {re}")
    except Exception as e:
        error_logger.error(f"An unexpected error occurred: {e}", exc_info=True)
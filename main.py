import logging
import sys
import argparse
import yaml
from typing import Optional, Union

sys.path.append("model")
sys.path.append("utils")

from model import Model
from utils import parse_text

logging.basicConfig(level=logging.DEBUG)


class ModelArguments:
    """
    Configuration class for model arguments.

    Args:
        model_name (Optional[str]): The name of the model. Default is "meta-llama/Llama-2-7b-hf".
        cache_dir (Optional[str]): The directory to cache the model. Default is None.
        r (Optional[int]): Parameter 'r' for the model. Default is 64.
        lora_alpha (Optional[float]): Alpha value for LoRA. Default is 32.
        lora_dropout (Optional[float]): Dropout value for LoRA. Default is 0.05.
        bits (Optional[int]): Number of bits. Default is 4.
        double_quant (Optional[bool]): Whether to double quantization or not. Default is True.
        quant_type (str): Type of quantization. Default is "nf4".
        trust_remote_code (Optional[bool]): Whether to trust remote code or not. Default is False.
        use_auth_token (Union[bool, str]): Authentication token. Default is False.
        compute_type (Optional[str]): Type of computation. Default is "bf16".
    """

    def __init__(
        self,
        model_name: Optional[str] = "codellama/CodeLlama-7b-hf",
        cache_dir: Optional[str] = None,
        r: Optional[int] = 64,
        lora_alpha: Optional[float] = 32,
        lora_dropout: Optional[float] = 0.05,
        bits: Optional[int] = 4,
        double_quant: Optional[bool] = True,
        quant_type: str = "nf4",
        trust_remote_code: Optional[bool] = False,
        use_auth_token: Union[bool, str] = False,
        compute_type: Optional[str] = "fp16",
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
        """
        Create an instance of ModelArguments from command line arguments.

        Args:
            args: Command line arguments.

        Returns:
            ModelArguments: An instance of ModelArguments.
        """
        return cls(**vars(args))

    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Create an instance of ModelArguments from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file.

        Returns:
            ModelArguments: An instance of ModelArguments.
        """
        with open(yaml_path, "r") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["model_arguments"]
        return cls(**yaml_args)


class DataTrainingArguments:
    """
    Configuration class for data training arguments.

    Args:
        dataset_name (Optional[str]): Name of the dataset. Default is "Dahoas/full-hh-rlhf".
        block_size (Optional[int]): Block size for the dataset. Default is 4096.
        multi_gpu (Optional[bool]): Whether to use multiple GPUs. Default is False.
        tensor_parallel (Optional[bool]): Whether to use tensor parallelism. Default is False.
        model_output_dir (Optional[str]): Output directory for the model. Default is "LLaMA/LoRA".
    """

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
        """
        Create an instance of DataTrainingArguments from command line arguments.

        Args:
            args: Command line arguments.

        Returns:
            DataTrainingArguments: An instance of DataTrainingArguments.
        """
        return cls(**vars(args))

    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Create an instance of DataTrainingArguments from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file.

        Returns:
            DataTrainingArguments: An instance of DataTrainingArguments.
        """
        with open(yaml_path, "r") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["data_training_arguments"]
        return cls(**yaml_args)


class Runner:
    """
    Main class to run the script.

    Methods:
        get_parser(): Get the argument parser.
        infer(model_config, prompt): Perform inference using the specified model configuration and prompt.
        finetune(model_config, data_training_config): Placeholder for the finetuning function.
        main(): Main entry point for the script.
    """

    def __init__(self) -> None:
        pass

    def get_parser(self):
        """
        Get the argument parser.

        Returns:
            argparse.ArgumentParser: The argument parser.
        """
        parser = argparse.ArgumentParser(description="Arguments")
        parser.add_argument(
            "--yaml_path",
            required=True,
            help="Path to the YAML file containing both sets of arguments.",
        )
        parser.add_argument(
            "--prompt_file",
            required=True,
            help="Path to the text file containing the prompt.",
        )
        parser.add_argument("--infer", action="store_true", help="Perform inference.")
        parser.add_argument(
            "--finetune", action="store_true", help="Perform finetuning."
        )
        return parser

    def infer(self, model_config, prompt):
        """
        Perform inference.

        Args:
            model_config: Model configuration.
            prompt (str): Input prompt for inference.

        Returns:
            str: Inference output.
        """
        try:
            model_infer = Model(model_config)
            model, tokenizer = model_infer.get_model_and_tokenizer()
            output = model_infer.model_inference(model, tokenizer, prompt)
            logging.info("Inference Output: %s", output)
            return output
        except Exception as e:
            logging.error("Error during inference: %s", e, exc_info=True)
            raise e

    def finetune(self, model_config, data_training_config):
        pass

    def main(self):
        """
        Main entry point for the script.
        """
        try:
            args = self.get_parser().parse_args()

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            debug_logger = logging.getLogger("debug_logger")
            error_logger = logging.getLogger("error_logger")

            logger.info("Starting the script.")

            model_arguments = ModelArguments.from_yaml(args.yaml_path)
            data_training_arguments = DataTrainingArguments.from_yaml(args.yaml_path)

            logger.info("Model Arguments:")
            logger.info(model_arguments.__dict__)

            logger.info("Data Training Arguments:")
            logger.info(data_training_arguments.__dict__)

            prompt = parse_text(args.prompt_file)

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

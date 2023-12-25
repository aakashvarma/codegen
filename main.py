import logging
import sys
import argparse
import yaml
from typing import Optional, Union

sys.path.append("model")
sys.path.append("utils")

from model import Model
from utils import parse_text

from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

class ModelConfiguration:
    """
    Configuration class for model arguments.

    Args:
        model_name (Optional[str]): The name of the model. Default is "codellama/CodeLlama-7b-hf".
        cache_dir (Optional[str]): The directory to cache the model. Default is None.
        r (Optional[int]): Parameter 'r' for the model. Default is 64.
        lora_alpha (Optional[float]): Alpha value for LoRA. Default is 32.
        lora_dropout (Optional[float]): Dropout value for LoRA. Default is 0.05.
        bits (Optional[int]): Number of bits. Default is 4.
        double_quant (Optional[bool]): Whether to double quantization or not. Default is True.
        quant_type (str): Type of quantization. Default is "nf4".
        trust_remote_code (Optional[bool]): Whether to trust remote code or not. Default is False.
        use_auth_token (Union[bool, str]): Authentication token. Default is False.
        compute_type (Optional[str]): Type of computation. Default is "fp16".
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
    def from_yaml(cls, yaml_path):
        """
        Create an instance of ModelConfiguration from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file.

        Returns:
            ModelConfiguration: An instance of ModelConfiguration.
        """
        with open(yaml_path, encoding="utf-8") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["model_config"]
        return cls(**yaml_args)


class TrainerConfiguration:
    """
    Configuration class for data training arguments.

    Args:
        dataset_name (Optional[str]): Name of the dataset. Default is "Dahoas/full-hh-rlhf".
        block_size (Optional[int]): Block size for the dataset. Default is 4096.
        multi_gpu (Optional[bool]): Whether to use multiple GPUs. Default is False.
        tensor_parallel (Optional[bool]): Whether to use tensor parallelism. Default is False.
        model_output_dir (Optional[str]): Output directory for the model. Default is "LLaMA/LoRA".
        compute_type (Optional[str]): Type of computation. Default is "fp16".
    """

    def __init__(
        self,
        dataset_name: Optional[str] = "Dahoas/full-hh-rlhf",
        block_size: Optional[int] = 4096,
        multi_gpu: Optional[bool] = False,
        tensor_parallel: Optional[bool] = False,
        model_output_dir: Optional[str] = "LLaMA/LoRA",
        per_device_train_batch_size: Optional[int] = 4,
        gradient_accumulation_steps: Optional[int] = 4,
        optim: Optional[str] = "paged_adamw_32bit",
        save_steps: Optional[int] = 100,
        logging_steps: Optional[int] = 10,
        learning_rate: Optional[float] = 2e-4,
        max_grad_norm: Optional[float] = 0.3,
        max_steps: Optional[int] = 100,
        warmup_ratio: Optional[float] = 0.03,
        lr_scheduler_type: Optional[str] = "constant",
        compute_type: Optional[str] = "fp16"
    ):
        self.dataset_name = dataset_name
        self.block_size = block_size
        self.multi_gpu = multi_gpu
        self.tensor_parallel = tensor_parallel
        self.model_output_dir = model_output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.compute_type = compute_type

    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Create an instance of TrainerConfiguration from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file.

        Returns:
            TrainerConfiguration: An instance of TrainerConfiguration.
        """
        with open(yaml_path, encoding="utf-8") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["trainer_config"]
        return cls(**yaml_args)

class FineTuneConfiguration:
    """
    Configuration class for fine-tuning arguments.

    Args:
        r (Optional[int]): Parameter 'r' for fine-tuning. Default is 16.
        lora_alpha (Optional[float]): Alpha value for LoRA during fine-tuning. Default is 32.
        lora_dropout (Optional[float]): Dropout value for LoRA during fine-tuning. Default is 0.05.
    """

    def __init__(
        self,
        r: Optional[int] = 16,
        lora_alpha: Optional[float] = 32,
        lora_dropout: Optional[float] = 0.05,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Create an instance of FineTuneConfiguration from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file.

        Returns:
            FineTuneConfiguration: An instance of FineTuneConfiguration.
        """
        with open(yaml_path, encoding="utf-8") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["finetune_config"]
        return cls(**yaml_args)


class Runner:
    """
    Main class to run the script.

    Methods:
        get_parser(): Get the argument parser.
        infer(model_config, prompt): Perform inference using the specified model configuration and prompt.
        finetune(model_config, trainer_config, finetune_config): Perform fine-tuning using the specified configurations.
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
        parser = argparse.ArgumentParser(description="Script Arguments")
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
            "--finetune", action="store_true", help="Perform fine-tuning."
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
            model = Model(model_config)
            output = model.infer_model(prompt)
            logging.info("Inference Done")
            return output
        except Exception as e:
            logging.error("Error during inference: %s", e, exc_info=True)
            raise e

    def finetune(self, model_config, trainer_config, finetune_config):
        """
        Perform fine-tuning.

        Args:
            model_config: Model configuration.
            trainer_config: Trainer configuration.
            finetune_config: Fine-tune configuration.
        """
        try:
            model = Model(model_config, trainer_config, finetune_config)
            model.finetune_model()
            logging.info("Fine-tuning completed.")
        except Exception as e:
            logging.error("Error during fine-tuning: %s", e, exc_info=True)
            raise e

    def main(self):
        """
        Main entry point for the script.
        """
        try:
            args = self.get_parser().parse_args()

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            error_logger = logging.getLogger("error_logger")

            logger.info("Starting the script.")

            model_config = ModelConfiguration.from_yaml(args.yaml_path)
            trainer_config = TrainerConfiguration.from_yaml(args.yaml_path)
            finetune_config = FineTuneConfiguration.from_yaml(args.yaml_path)

            logger.info("Model Configuration:")
            logger.info(model_config.__dict__)

            logger.info("LLMTrainer Configuration:")
            logger.info(trainer_config.__dict__)

            logger.info("FineTune Configuration:")
            logger.info(finetune_config.__dict__)

            prompt = parse_text(args.prompt_file)

            if args.infer:
                result = self.infer(model_config, prompt)
                logger.info("Script completed successfully with result: %s", result)
            elif args.finetune:
                self.finetune(model_config, trainer_config, finetune_config)
                logger.info("Script completed fine-tuning successfully.")

        except ValueError as ve:
            error_logger.error("ValueError: %s", ve)
        except RuntimeError as re:
            error_logger.error("RuntimeError: %s", re)
        except Exception as e:
            error_logger.error("An unexpected error occurred: %s", e, exc_info=True)


if __name__ == "__main__":
    Runner().main()

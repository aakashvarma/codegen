import argparse
import logging
import os
import pickle
import sys
from datetime import datetime

import yaml

sys.path.append("model")
sys.path.append("utils")

from model import Model
from utils import parse_prompt, extract_question_context

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"script_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)
error_logger = logging.getLogger(__name__ + "_error")


class ModelConfiguration:
    def __init__(
            self,
            model_name="codellama/CodeLlama-7b-hf",
            cache_dir=None,
            r=64,
            lora_alpha=32.0,
            lora_dropout=0.05,
            bits=4,
            double_quant=True,
            quant_type="nf4",
            trust_remote_code=False,
            use_auth_token=False,
            compute_type="fp16",
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
        with open(yaml_path, encoding="utf-8") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["model_config"]
        return cls(**yaml_args)

class TrainerConfiguration:
    def __init__(
            self,
            dataset_name="b-mc2/sql-create-context",
            block_size=512,
            multi_gpu=False,
            tensor_parallel=False,
            model_output_dir="__run.default",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=100,
            logging_steps=10,
            learning_rate=0.0002,
            max_grad_norm=0.3,
            max_steps=100,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            compute_type="fp16",
            num_train_epochs=1,
            evaluation_strategy='steps'
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
        self.num_train_epochs = num_train_epochs
        self.evaluation_strategy = evaluation_strategy

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, encoding="utf-8") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["trainer_config"]
        return cls(**yaml_args)


class FineTuneConfiguration:
    def __init__(
            self,
            r=16,
            lora_alpha=32.0,
            lora_dropout=0.05,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, encoding="utf-8") as yaml_file:
            yaml_args = yaml.safe_load(yaml_file)["finetune_config"]
        return cls(**yaml_args)


class Runner:
    def __init__(self) -> None:
        pass

    def infer(self, model_config, context, question, answer, model_path, model_with_adapter, merge_model, is_verif=False, val_output_filepath=None):
        try:
            logging.info("Inference started.")
            model = Model(model_config)
            output = model.infer_model(context, question, answer, model_path, model_with_adapter, merge_model, is_verif, val_output_filepath)
            logging.info("Inference Done")
            return output
        except Exception as e:
            logging.error("Error during inference: %s", e, exc_info=True)
            raise e

    def finetune(self, model_config, trainer_config, finetune_config, model_path, model_with_adapter, merge_model):
        try:
            logging.info("Fine-tuning started.")
            model = Model(model_config, trainer_config, finetune_config)
            model.finetune_model(model_path, model_with_adapter, merge_model)
            logging.info("Fine-tuning completed.")
        except Exception as e:
            logging.error("Error during fine-tuning: %s", e, exc_info=True)
            raise e

    def validate(self, model_config, validation_dir, model_path, model_with_adapter, merge_model):
        try:
            logging.info("Validation started.")
            val_input_filename = "val_data.pkl"
            val_output_filename = "val_output.pkl"
            val_input_filepath = os.path.join(validation_dir, val_input_filename)
            val_output_filepath = os.path.join(validation_dir, val_output_filename)

            if not os.path.exists(val_output_filepath):
                with open(val_output_filepath, 'w'):
                    pass  # This will create an empty file

            with open(val_input_filepath, "rb") as file:
                loaded_data = pickle.load(file)
        except Exception as e:
            logging.error("Error while loading pickle file: %s", e, exc_info=True)
            raise e
        try:
            val_context = loaded_data["context"]
            val_question = loaded_data["question"]
            val_answer = loaded_data["answer"]

            self.infer(model_config, val_context, val_question, val_answer, model_path, model_with_adapter, merge_model, True, val_output_filepath)

            logging.info("Validation completed.")
        except Exception as e:
            logging.error("Error during model validation: %s", e, exc_info=True)
            raise e

    def validate_args(self, args):
        logger.info("Validating arguments.")
        if args.merge_adapter and not args.model_path:
            raise argparse.ArgumentError("--merge_adapter requires --model_path to be provided.")
        if args.model_with_adapter and not args.model_path:
            raise argparse.ArgumentError("--model_with_adapter requires --model_path to be provided.")
        if args.merge_adapter and args.model_with_adapter:
            raise argparse.ArgumentError("--merge_adapter and --model_with_adapter cannot be provided together. --model_with_adapter should be used only for inference.")
        if args.finetune and args.model_path:
            raise argparse.ArgumentError("--finetune and --model_path cannot be passed together")

    def get_parser(self):
        parser = argparse.ArgumentParser(description="Script Arguments")
        parser.add_argument(
            "--model_path",
            help="Path to the model file.",
        )
        parser.add_argument(
            "--model_yaml",
            required=True,
            help="Path to the YAML file containing model configuration.",
        )
        parser.add_argument(
            "--trainer_yaml",
            help="Path to the YAML file containing trainer configuration.",
        )
        parser.add_argument(
            "--finetune_yaml",
            help="Path to the YAML file containing fine-tune configuration.",
        )
        parser.add_argument(
            "--prompt_file",
            help="Path to the text file containing the prompt.",
        )
        parser.add_argument(
            "--validation_dir",
            help="Path to the pickle file containing the validation data.",
        )
        parser.add_argument(
            "--infer",
            action="store_true",
            help="Perform inference.",
            default=False
        )
        parser.add_argument(
            "--finetune",
            action="store_true",
            help="Perform fine-tuning.",
            default=False
        )
        parser.add_argument(
            "--validate",
            action="store_true",
            help="Perform validation.",
            default=False
        )
        parser.add_argument(
            "--model_with_adapter",
            action="store_true",
            help="If True and model path is passed, then model path should have the adapter details. Else if False, then it is assumed that the model is a merged model / base model.",
            default=False
        )
        parser.add_argument(
            "--merge_adapter",
            action="store_true",
            help="When set to True, the adapter is merged to the model present in the model path.",
            default=False
        )
        return parser

    def main(self):
        try:
            args = self.get_parser().parse_args()
            self.validate_args(args)

            logger.info("Starting the script.")

            model_config = ModelConfiguration.from_yaml(args.model_yaml)

            if args.infer:
                # For inference, only model YAML is required
                logger.info("Model Configuration:")
                logger.info(model_config.__dict__)

                prompt = parse_prompt(args.prompt_file)
                question, context = extract_question_context(prompt)

                result = self.infer(model_config, context, question,None, args.model_path, args.model_with_adapter, args.merge_model)
                logger.info("Script completed successfully with result: %s", result)
            elif args.finetune:
                # For fine-tuning, all three YAML files are required
                if not args.trainer_yaml or not args.finetune_yaml:
                    raise ValueError("Both --trainer_yaml and --finetune_yaml are required for fine-tuning.")

                trainer_config = TrainerConfiguration.from_yaml(args.trainer_yaml)
                finetune_config = FineTuneConfiguration.from_yaml(args.finetune_yaml)

                logger.info("Model Configuration:")
                logger.info(model_config.__dict__)

                logger.info("LLMTrainer Configuration:")
                logger.info(trainer_config.__dict__)

                logger.info("FineTune Configuration:")
                logger.info(finetune_config.__dict__)

                self.finetune(model_config, trainer_config, finetune_config,
                              args.model_path, args.model_with_adapter, args.merge_model)
                logger.info("Script completed fine-tuning successfully.")
            elif args.validate:
                logger.info("Model Configuration:")
                logger.info(model_config.__dict__)

                self.validate(model_config, args.validation_dir, args.model_path, args.model_with_adapter, args.merge_model)
                logger.info("Script completed successfully")

        except ValueError as ve:
            error_logger.error("ValueError: %s", ve)
        except RuntimeError as re:
            error_logger.error("RuntimeError: %s", re)
        except ImportError as e:
            error_logger.error("An unexpected error occurred: %s", e, exc_info=True)

if __name__ == "__main__":
    Runner().main()

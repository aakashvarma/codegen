import logging
import sys
import re

import torch
import pickle
from tqdm import tqdm

from model_operators.finetune import Quantizer, FineTuner
from trainer.trainer import LLMTrainer
from utils.utils import extract_sql_output

sys.path.append("../utils")
sys.path.append("../model_operators")
sys.path.append("../trainer")


class Model:
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
        return f"Model Config: {self.model_config}"

    def get_inference_model_and_tokenizer(self):
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
        try:
            logging.info("Setting up model for fine-tuning.")
            finetuner = FineTuner(self.model_config, self.finetune_config)
            self.model, self.tokenizer = finetuner.model_setup()
            return self.model, self.tokenizer

        except Exception as e:
            error_message = f"Error on setting up the model for fine-tuning: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message) from e

    def infer_model(self, context, question, answer, is_verif, val_output_filepath):
        try:
            self.get_inference_model_and_tokenizer()
            logging.info("Start model inference.")
            sql_output_arr = []
            real_output_arr = []
            prompts = []
            answers = []
            full_prompt = (
"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.
You must output the SQL query that answers the question.

### Input:
{}

### Context:
{}

### Response:
"""
            )
            if(is_verif):
                mini_batch = 4
                logging.info("Start tokenizing prompts.")
                for k in tqdm(range(0, 1000, mini_batch), desc="Outer Loop"):
                    for i in tqdm(range(k, k + mini_batch), desc="Inner Loop", leave=False):
                        prompts.append(full_prompt.format(question[i], context[i]))
                        answers.append(answer[i])

                    model_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

                    logging.info("Start generating outputs.")
                    self.model.eval()

                    with torch.no_grad():
                        generated_tokens = self.model.generate(
                            **model_inputs, max_new_tokens=100
                        )
                        decoded_output = self.tokenizer.batch_decode(
                            generated_tokens, skip_special_tokens=True
                        )
                        sql_output_arr.append(decoded_output)
                        real_output_arr.append(answers)

                    prompts = []
                    answers = []

                with open(val_output_filepath, 'wb') as file:
                    pickle.dump((sql_output_arr, real_output_arr), file)
            else:
                logging.info("Start tokenizing prompts.")

                prompt = (full_prompt.format(question, context))
                model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

                logging.info("Start generating outputs.")
                self.model.eval()

                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **model_inputs, max_new_tokens=100
                    )
                    decoded_output = self.tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    sql_output = extract_sql_output(decoded_output)
                    print("Output: ", sql_output)

        except Exception as e:
            error_message = f"Error during model inference: {e}"
            logging.error(error_message, exc_info=True)
            raise RuntimeError(error_message) from e

    def finetune_model(self):
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

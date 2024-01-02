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
    """
    Class for working with a pretrained language model and performing inference.

    Args:
        model_config: Model configuration object.
        trainer_config: Trainer configuration object.
        finetune_config: Fine-tune configuration object.

    Methods:
        __init__(model_config, trainer_config=None, finetune_config=None): Initialize the Model class with attributes.
        __str__(): Provide a string representation of the Model instance.
        get_inference_model_and_tokenizer(): Download a pretrained model and tokenizer based on the given base model ID using the selected adapter for inference.
        get_finetuning_model_and_tokenizer(): Download a pretrained model and tokenizer based on the given base model ID using the selected adapter for fine-tuning.
        infer_model(prompt): Perform model inference on the provided prompt.
        finetune_model(): Fine-tune the model with the given training and validation data.

    Attributes:
        model_config: Model configuration object.
        trainer_config: Trainer configuration object.
        finetune_config: Fine-tune configuration object.
        model: Pretrained language model.
        tokenizer: Tokenizer associated with the model.
    """

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
        """
        Provide a string representation of the Model instance.

        Returns:
            str: String representation of the Model instance.
        """
        return f"Model Config: {self.model_config}"

    def get_inference_model_and_tokenizer(self):
        """
        Download a pretrained model and tokenizer based on the given base model ID using the selected adapter for inference.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
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
        """
        Download a pretrained model and tokenizer based on the given base model ID using the selected adapter for fine-tuning.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        try:
            logging.info("Setting up model for fine-tuning.")
            finetuner = FineTuner(self.model_config, self.finetune_config)
            self.model, self.tokenizer = finetuner.model_setup()
            return self.model, self.tokenizer

        except Exception as e:
            error_message = f"Error on setting up the model for fine-tuning: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message) from e

#     def infer_model(self, context, question, answer):
#         """
#         Perform model inference on the provided prompt.
#
#         Args:
#             prompt (str): The input prompt for inference.
#
#         Returns:
#             str: Inference output.
#         """
#         try:
#             self.get_inference_model_and_tokenizer()
#             logging.info("Start model inference.")
#             sql_output_arr = []
#             real_output_arr = []
#             model_inputs = []
#
#             logging.info("Start tokenizing prompts.")
#             for i in range(0, 1000):
#                 full_prompt = (
# """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.
# You must output the SQL query that answers the question.
#
# ### Input:
# {}
#
# ### Context:
# {}
#
# ### Response:
# """
#                 )
#                 prompt = full_prompt.format(question[i], context[i])
#                 model_inputs.append(self.tokenizer(prompt, return_tensors="pt").to("cuda"))
#
#             logging.info("Start generating outputs.")
#             self.model.eval()
#
#             with torch.no_grad():
#                 for i in range(0, len(model_inputs)):
#                     generated_tokens = self.model.generate(
#                         **model_inputs[i], max_new_tokens=50
#                     )[0]
#                     decoded_output = self.tokenizer.decode(
#                         generated_tokens, skip_special_tokens=True
#                     )
#                     sql_output_arr.append(decoded_output)
#                     real_output_arr.append(answer[i])
#                     print(i, ": ", decoded_output)
#                     # match = re.search(r'### Response:(.+?)(?:\n|$)', decoded_output, re.DOTALL)
#                     # if match:
#                     #     result_line = match.group(1).strip()
#                     #     sql_output_arr.append(result_line)
#                     #     real_output_arr.append(answer[i])
#                     #     print(i,  ": ", result_line)
#                     # else:
#                     #     error_message = "Output ### Response: not found."
#                     #     logging.error(error_message)
#                     #     raise ValueError(error_message)
#
#             with open('output_data.pkl', 'wb') as file:
#                 pickle.dump((sql_output_arr, real_output_arr), file)
#
#         except Exception as e:
#             error_message = f"Error during model inference: {e}"
#             logging.error(error_message, exc_info=True)
#             raise RuntimeError(error_message) from e

    def infer_model(self, context, question, answer, is_verif):
        """
        Perform model inference on the provided prompt.

        Args:
            prompt (str): The input prompt for inference.

        Returns:
            str: Inference output.
        """
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

                with open('output_data.pkl', 'wb') as file:
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
        """
        Fine-tune the model with the given training and validation data.

        Raises:
            RuntimeError: If an error occurs during fine-tuning.
        """
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

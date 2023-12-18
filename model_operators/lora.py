# import logging

# from datetime import datetime
# import os
# import sys
# import torch

# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForSeq2Seq,
#     BitsAndBytesConfig,
# )


# from datasets import load_dataset

# dataset = load_dataset("b-mc2/sql-create-context", split="train")

# train_dataset = dataset.train_test_split(test_size=0.1)["train"]
# eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

# base_model = "codellama/CodeLlama-7b-hf"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     base_model, quantization_config=bnb_config, trust_remote_code=True
# )
# model.config.use_cache = False

# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

# eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

# You must output the SQL query that answers the question.
# ### Input:
# Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

# ### Context:
# CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

# ### Response:
# """
# # {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}
# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# model.eval()
# with torch.no_grad():
#     print(
#         tokenizer.decode(
#             model.generate(**model_input, max_new_tokens=100)[0],
#             skip_special_tokens=True,
#         )
#     )


# tokenizer.add_eos_token = True
# tokenizer.pad_token_id = 0
# tokenizer.padding_side = "left"


# def tokenize(prompt):
#     result = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=512,
#         padding=False,
#         return_tensors=None,
#     )

#     # "self-supervised learning" means the labels are also the inputs:
#     result["labels"] = result["input_ids"].copy()

#     return result


# def generate_and_tokenize_prompt(data_point):
#     full_prompt = f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

# You must output the SQL query that answers the question.

# ### Input:
# {data_point["question"]}

# ### Context:
# {data_point["context"]}

# ### Response:
# {data_point["answer"]}
# """
#     return tokenize(full_prompt)


# tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
# tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)


# from peft import LoraConfig, get_peft_model

# lora_alpha = 16
# lora_dropout = 0.1
# lora_r = 64

# peft_config = LoraConfig(
#     lora_alpha=lora_alpha,
#     lora_dropout=lora_dropout,
#     r=lora_r,
#     bias="none",
#     task_type="CAUSAL_LM",
# )


# from transformers import TrainingArguments

# output_dir = "./results"
# per_device_train_batch_size = 4
# gradient_accumulation_steps = 4
# optim = "paged_adamw_32bit"
# save_steps = 100
# logging_steps = 10
# learning_rate = 2e-4
# max_grad_norm = 0.3
# max_steps = 100
# warmup_ratio = 0.03
# lr_scheduler_type = "constant"

# training_arguments = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=per_device_train_batch_size,
#     gradient_accumulation_steps=gradient_accumulation_steps,
#     optim=optim,
#     save_steps=save_steps,
#     logging_steps=logging_steps,
#     learning_rate=learning_rate,
#     fp16=True,
#     max_grad_norm=max_grad_norm,
#     max_steps=max_steps,
#     warmup_ratio=warmup_ratio,
#     group_by_length=True,
#     lr_scheduler_type=lr_scheduler_type,
# )


# model.add_adapter(peft_config)

# trainer = Trainer(
#     model=model,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_val_dataset,
#     args=training_arguments,
#     data_collator=DataCollatorForSeq2Seq(
#         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
#     ),
# )

# trainer.train()


import logging
from model_operators.finetune import FineTuning

import torch
from transformers import BitsAndBytesConfig


class LoRA(FineTuning):
    def __init__(self):
        super().__init__()

    def finetuning_model_setup(self, base_model):
        """
        Set up QLoRA configuration.

        Args:
            base_model (str): The base model ID.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        logging.info("Setting up QLoRA configuration.")
        try:
            model_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    )
            model = self.get_model(model_config, base_model)
            tokenizer = self.get_tokenizer(base_model)

            logging.info("QLoRA configuration successful.")

        except Exception as e:
            error_message = f"Error in setting up QLoRA configuration: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message)

        return model, tokenizer

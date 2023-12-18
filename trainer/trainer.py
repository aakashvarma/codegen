import logging
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset


class Trainer:

	def __init__(self, model, tokenizer, trainer_config):
		self.model = model
		self.trainer_config = trainer_config
		self.tokenizer = tokenizer

	def get_dataset(self):
		dataset = load_dataset(self.trainer_config.dataset_name, split="train")

		train_dataset = dataset.train_test_split(test_size=0.1)["train"]
		eval_dataset = dataset.train_test_split(test_size=0.1)["test"]
		return train_dataset, eval_dataset

	
	def tokenize(self, prompt):
		result = self.tokenizer(
			prompt,
			truncation=True,
			max_length=self.trainer_config.block_size,
			padding=False,
			return_tensors=None,
		)
		result["labels"] = result["input_ids"].copy()
		return result


	def generate_and_tokenize_prompt(self, data_point):
		full_prompt =   f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

						You must output the SQL query that answers the question.

						### Input:
						{data_point["question"]}

						### Context:
						{data_point["context"]}

						### Response:
						{data_point["answer"]}
						"""
		return self.tokenize(full_prompt)



	def get_trianer(self):
		self.tokenizer.add_eos_token = True
		self.tokenizer.pad_token_id = 0
		self.tokenizer.padding_side = "left"

		train_dataset, eval_dataset = self.get_dataset()

		tokenized_train_dataset = train_dataset.map(self.generate_and_tokenize_prompt)
		tokenized_val_dataset = eval_dataset.map(self.generate_and_tokenize_prompt)

		training_arguments = TrainingArguments(
			output_dir = self.trainer_config.model_output_dir,
			per_device_train_batch_size = self.trainer_config.per_device_train_batch_size,
			gradient_accumulation_steps = self.trainer_config.gradient_accumulation_steps,
			optim = self.trainer_config.optim,
			save_steps = self.trainer_config.save_steps,
			logging_steps = self.trainer_config.logging_steps,
			learning_rate = self.trainer_config.learning_rate,
			fp16 = self.trainer_config.compute_type == "fp16",
			bf16 = self.trainer_config.compute_type == "bf16",
			max_grad_norm = self.trainer_config.max_grad_norm,
			max_steps = self.trainer_config.max_steps,
			warmup_ratio = self.trainer_config.warmup_ratio,
			group_by_length = True,
			lr_scheduler_type = self.trainer_config.lr_scheduler_type,
		)

		trainer = Trainer(
			model = self.model,
			train_dataset = tokenized_train_dataset,
			eval_dataset = tokenized_val_dataset,
			args = training_arguments,
			data_collator = DataCollatorForSeq2Seq(
				self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
			),
		)

		return trainer


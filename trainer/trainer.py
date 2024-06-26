import logging
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
from datasets import load_dataset
import pickle
import os
import torch
import numpy as np
import evaluate


class LLMTrainer:
    """
    A class for training a Language Model (LM) on a custom dataset using the transformers library.

    Attributes:
        model: The language model to be trained.
        trainer_config: Configuration for the training process.
        tokenizer: The tokenizer associated with the language model.

    Methods:
        __init__(self, model, tokenizer, trainer_config):
            Initializes the LLMTrainer with the provided model, tokenizer, and training configurations.

        get_dataset(self):
            Loads and splits the training dataset into training and evaluation sets.

        tokenize(self, prompt):
            Tokenizes a given prompt using the tokenizer and adds labels for LM training.

        generate_and_tokenize_prompt(self, data_point):
            Generates a prompt for the LM using the given data point and tokenizes it.

        get_trainer(self):
            Configures and returns a Trainer object for training the language model.

    """

    def __init__(self, model, tokenizer, trainer_config):
        """
        Initializes the LLMTrainer.

        Args:
            model: The language model to be trained.
            tokenizer: The tokenizer associated with the language model.
            trainer_config: Configuration for the training process.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_config = trainer_config

    def get_dataset(self):
        """
        Loads and splits the training dataset into training and evaluation sets.

        Returns:
            tuple: A tuple containing the training and evaluation datasets.
        """
        dataset = load_dataset(self.trainer_config.dataset_name, split="train")
        # 80% train, 20% test
        train_test_dataset = dataset.train_test_split(test_size=0.2)
        # Split the 20% test into half test, half valid
        test_valid = train_test_dataset['test'].train_test_split(test_size=0.5)

        train_dataset = train_test_dataset['train']
        eval_dataset = test_valid['train']
        validation_dataset = test_valid['test']

        val_context = validation_dataset['context']
        val_question = validation_dataset['question']
        val_answer = validation_dataset['answer']

        # Dumping the validation lists to a file using pickle
        try:
            val_data_filename = "val_data.pkl"
            val_file_path = os.path.join(self.trainer_config.model_output_dir, val_data_filename)

            if not os.path.exists(self.trainer_config.model_output_dir):
                os.makedirs(self.trainer_config.model_output_dir)

            if not os.path.exists(val_file_path):
                with open(val_file_path, 'w'):
                    pass  # This will create an empty file

            with open(val_file_path, "wb") as file:
                data = {"context": val_context, "question": val_question, "answer": val_answer}
                pickle.dump(data, file)
        except Exception as e:
            logging.error("Error while dumping pickle file: %s", e, exc_info=True)
            raise e

        return train_dataset, validation_dataset

    def get_gptq_dataset(self, num_samples):
        dataset = load_dataset(self.trainer_config.dataset_name, split="train")
        shuffled_dataset = dataset.shuffle(seed=42)
        small_dataset = shuffled_dataset.select(range(num_samples))
        def generate_and_tokenize_prompt(data_point):
            prompts = []
            for i in data_point:
                full_prompt = (
f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.
You must output the SQL query that answers the question.

### Input:
{i["question"]}

### Context:
{i["context"]}

### Response:
{i["answer"]}
"""
                )
                prompts.append(full_prompt)
            return prompts

        dataset = generate_and_tokenize_prompt(small_dataset)
        return dataset

    def tokenize(self, prompt):
        """
        Tokenizes a given prompt using the tokenizer and adds labels for LM training.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            dict: A dictionary containing tokenized input and labels.
        """
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
        """
        Generates a prompt for the LM using the given data point and tokenizes it.

        Args:
            data_point (dict): The data point containing question, context, and answer.

        Returns:
            dict: A dictionary containing tokenized input and labels for the generated
                  prompt.
        """
        full_prompt = (
f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.
You must output the SQL query that answers the question.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{data_point["answer"]}
"""
        )
        return self.tokenize(full_prompt)

    def get_trainer(self):
        """
        Configures and returns a Trainer object for training the language model.

        Returns:
            Trainer: The configured Trainer object.
        """
        self.configure_tokenizer()
        train_dataset, eval_dataset = self.get_dataset()
        tokenized_train_dataset = train_dataset.map(self.generate_and_tokenize_prompt)
        tokenized_val_dataset = eval_dataset.map(self.generate_and_tokenize_prompt)

        training_arguments = self.configure_training_arguments()

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            metric = evaluate.load("rouge")

            if isinstance(preds, tuple):
                preds = preds[0]
            for idx in range(len(preds)):
                for idx2 in range(len(preds[idx])):
                    if preds[idx][idx2] == -100:
                        preds[idx][idx2] = 50256
                    if labels[idx][idx2] == -100:
                        labels[idx][idx2] = 50256

            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

            # The results of the Rouge metric are then multiplied by 100 and rounded to four decimal places.
            result = {k: round(v * 100, 4) for k, v in result.items()}

            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)

            print(result)

            return result


        def preprocess_logits_for_metrics(logits, labels):
            pred_ids = torch.argmax(logits, dim=-1)
            return pred_ids, labels

        trainer = Trainer(
            model=self.model,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            args=training_arguments,
            data_collator=self.configure_data_collator(),
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        return trainer

    def configure_tokenizer(self):
        """Configures the tokenizer for the LM training."""
        self.tokenizer.add_eos_token = True
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

    def configure_training_arguments(self):
        """Configures the training arguments for the Trainer."""
        return TrainingArguments(
            output_dir=self.trainer_config.model_output_dir,
            per_device_train_batch_size=self.trainer_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.trainer_config.gradient_accumulation_steps,
            optim=self.trainer_config.optim,
            save_steps=self.trainer_config.save_steps,
            logging_steps=self.trainer_config.logging_steps,
            learning_rate=self.trainer_config.learning_rate,
            fp16=self.trainer_config.compute_type == "fp16",
            bf16=self.trainer_config.compute_type == "bf16",
            max_grad_norm=self.trainer_config.max_grad_norm,
            # max_steps=100,
            num_train_epochs=self.trainer_config.num_train_epochs,
            warmup_ratio=self.trainer_config.warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=self.trainer_config.lr_scheduler_type,
            evaluation_strategy=self.trainer_config.evaluation_strategy,
            eval_steps=self.trainer_config.logging_steps,
        )

    def configure_data_collator(self):
        """Configures the data collator for the Trainer."""
        return DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

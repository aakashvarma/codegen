import logging

import torch
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os


class Quantizer:
    def __init__(self, model_config):
        self.model_config = model_config

    def get_model(self, model_path, model_with_adapter, merge_model):
        device_map = "auto"
        compute_dtype = (
            torch.float16
            if self.model_config.compute_type == "fp16"
            else (
                torch.bfloat16
                if self.model_config.compute_type == "bf16"
                else torch.float32
            )
        )
        if merge_model:
            try:
                directory, filename = os.path.split(model_path)
                directories = directory.split(os.path.sep)
                directories[-1] += "_merged_model"
                merged_model_path = os.path.join(os.path.sep.join(directories), filename)

                logging.info("Picking the pre-tuned model from the path to be merged: {}".format(model_path))

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_config.model_name,
                    trust_remote_code=True,
                    return_token_type_ids=False
                )
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    return_dict=True,
                    torch_dtype=compute_dtype,
                    device_map="cuda",
                )
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(merged_model_path, safe_serialization=True)
                tokenizer.save_pretrained(merged_model_path)

                model = merged_model

                logging.info("Model adapter merged and saved to the path: {}".format(merged_model_path))
            except Exception as e:
                error_message = "Error occurred in get_model() of quantize.py while merging_model."
                logging.error(error_message)
                raise ValueError(error_message)

        # execute model present in the path
        elif merge_model is False and model_with_adapter is False and model_path is not None:
            try:
                if "gptq_quantized_model" in model_path:
                    logging.info("Loading GPTQ model from path: {}".format(model_path))
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path
                    )
                else:
                    logging.info("Loading model from path (merged model): {}".format(model_path))
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        load_in_4bit=self.model_config.bits == 4,
                        load_in_8bit=self.model_config.bits == 8,
                        device_map=device_map,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=self.model_config.bits == 4,
                            load_in_8bit=self.model_config.bits == 8,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_use_double_quant=self.model_config.double_quant,
                            bnb_4bit_quant_type=self.model_config.quant_type,
                        ),
                        torch_dtype=(
                            torch.float32
                            if self.model_config.compute_type == "fp16"
                            else (
                                torch.bfloat16
                                if self.model_config.compute_type == "bf16"
                                else torch.float32
                            )
                        ),
                        trust_remote_code=self.model_config.trust_remote_code,
                        use_auth_token=self.model_config.use_auth_token,
                    )
            except Exception as e:
                error_message = "Error in loading model without adapter."
                logging.error(error_message)
                raise ValueError(error_message)

        else:
            if (self.model_config.model_name == "meta-llama/Llama-2-7b-hf"):
                from huggingface_hub import login
                login()
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name,
                cache_dir=self.model_config.cache_dir,
                load_in_4bit=self.model_config.bits == 4,
                load_in_8bit=self.model_config.bits == 8,
                device_map=device_map,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=self.model_config.bits == 4,
                    load_in_8bit=self.model_config.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=self.model_config.double_quant,
                    bnb_4bit_quant_type=self.model_config.quant_type,
                ),
                torch_dtype=(
                    torch.float32
                    if self.model_config.compute_type == "fp16"
                    else (
                        torch.bfloat16
                        if self.model_config.compute_type == "bf16"
                        else torch.float32
                    )
                ),
                trust_remote_code=self.model_config.trust_remote_code,
                use_auth_token=self.model_config.use_auth_token,
            )
            model.config.use_cache = False

            if model_with_adapter:
                logging.info("Picking the pre-tuned model from the path: {}".format(model_path))
                logging.info("This model has a LoRA adapter.")
                try:
                    model = PeftModel.from_pretrained(model, model_path)
                except Exception as e:
                    error_message = "Error in loading model with adapter."
                    logging.error(error_message)
                    raise ValueError(error_message)

        return model

    def get_tokenizer(self):
        """
        Get the tokenizer for the fine-tuning model.

        Returns:
            transformers.AutoTokenizer: The fine-tuning model tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        if tokenizer.pad_token is None:
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def model_setup(self, model_path, model_with_adapter, merge_model):
        """
        Set up Finetune configuration.

        Returns:
            tuple: A tuple containing the configured model and tokenizer.
        """
        logging.info("Setting up Model configuration.")
        try:
            model = self.get_model(model_path, model_with_adapter, merge_model)
            tokenizer = self.get_tokenizer()

            logging.info("Model configuration successful.")

        except Exception as e:
            logging.error(f"Error in setting up Model configuration: {e}")
            raise

        return model, tokenizer

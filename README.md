# codegen

## Introduction

This documentation provides detailed information on the installation, usage, configuration, error handling, contributing guidelines, and license information for the script designed for model inference and finetuning. The script is tailored to work with a specific model and offers customizable behavior through YAML configuration files.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:aakashvarma/codegen.git
   cd codegen
   ```

2. **Install Dependencies:**

   ```bash
   pip install git+https://github.com/huggingface/transformers.git@main accelerate bitsandbytes
   pip install git+https://github.com/huggingface/peft.git@4c611f4
   pip install datasets==2.10.1
   pip install wandb
   pip install scipy
   pip install trl
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Example 1: Inference

Perform model inference using the following command:

```bash
python script.py --model_yaml config.yaml --prompt_file prompt.txt --infer
```

### Example 2: Finetuning

Perform model finetuning with the following command:

```bash
python script_name.py --model_yaml path/to/config.yaml --trainer_yaml trainer_config.yaml --finetune_yaml finetune_config.yaml --finetune
```

## Configuration

### ModelConfiguration

Configuration class for model arguments.

#### Arguments:

- `model_name` (Optional[str]): The name of the model. Default is "codellama/CodeLlama-7b-hf".
- `pretrained_model_dir` (Optional[str]): Pre-trained model directory path. Default is None.
- `cache_dir` (Optional[str]): The directory to cache the model. Default is None.
- `r` (Optional[int]): Parameter 'r' for the model. Default is 64.
- `lora_alpha` (Optional[float]): Alpha value for LoRA. Default is 32.
- `lora_dropout` (Optional[float]): Dropout value for LoRA. Default is 0.05.
- `bits` (Optional[int]): Number of bits. Default is 4.
- `double_quant` (Optional[bool]): Whether to double quantization or not. Default is True.
- `quant_type` (str): Type of quantization. Default is "nf4".
- `trust_remote_code` (Optional[bool]): Whether to trust remote code or not. Default is False.
- `use_auth_token` (Union[bool, str]): Authentication token. Default is False.
- `compute_type` (Optional[str]): Type of computation. Default is "fp16".

#### Methods:

- `from_yaml(yaml_path)`: Create an instance of ModelConfiguration from a YAML file.

### TrainerConfiguration

Configuration class for data training arguments.

#### Arguments:

- `dataset_name` (Optional[str]): Name of the dataset. Default is "b-mc2/sql-create-context".
- `block_size` (Optional[int]): Block size for the dataset. Default is 512.
- `multi_gpu` (Optional[bool]): Whether to use multiple GPUs. Default is False.
- `tensor_parallel` (Optional[bool]): Whether to use tensor parallelism. Default is False.
- `model_output_dir` (Optional[str]): Output directory for the model. Default is "__run.default".
- `per_device_train_batch_size` (Optional[int]): Batch size per device during training. Default is 4.
- `gradient_accumulation_steps` (Optional[int]): Number of gradient accumulation steps. Default is 4.
- `optim` (Optional[str]): Optimization algorithm. Default is "paged_adamw_32bit".
- `save_steps` (Optional[int]): Number of steps before saving a checkpoint. Default is 100.
- `logging_steps` (Optional[int]): Number of steps before logging. Default is 10.
- `learning_rate` (Optional[float]): Learning rate for training. Default is 0.0002.
- `max_grad_norm` (Optional[float]): Maximum gradient norm for gradient clipping. Default is 0.3.
- `max_steps` (Optional[int]): Maximum number of training steps. Default is 100.
- `warmup_ratio` (Optional[float]): Warmup ratio for the learning rate schedule. Default is 0.03.
- `lr_scheduler_type` (Optional[str]): Type of learning rate scheduler. Default is "constant".
- `compute_type` (Optional[str]): Type of computation. Default is "fp16".

#### Methods:

- `from_yaml(yaml_path)`: Create an instance of TrainerConfiguration from a YAML file.

### FineTuneConfiguration

Configuration class for fine-tuning arguments.

#### Arguments:

- `r` (Optional[int]): Parameter 'r' for fine-tuning. Default is 16.
- `lora_alpha` (Optional[float]): Alpha value for LoRA during fine-tuning. Default is 32.
- `lora_dropout` (Optional[float]): Dropout value for LoRA during fine-tuning. Default is 0.05.

#### Methods:

- `from_yaml(yaml_path)`: Create an instance of FineTuneConfiguration from a YAML file.

## Error Handling

The script provides comprehensive error handling for various exceptions. In case of an error, the script logs details and exits gracefully.

## Contributing

Contributions to the project are welcomed. If you encounter a bug or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [Apache-2.0 license](LICENSE). Feel free to use and modify the code according to the terms of the license.
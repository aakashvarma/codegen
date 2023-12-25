# codegen

## Introduction

This documentation provides information on the usage and configuration of the script for model inference and finetuning. The script is designed to work with a specific model, and its behavior can be customized through configuration files in YAML format.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Error Handling](#error-handling)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:aakashvarma/codegen.git
   cd codegen
   ```

2. Install the required dependencies:

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

The script is designed to perform either inference or finetuning based on the provided command-line arguments. Follow the instructions below to use the script:

```bash
python script.py --yaml_path <path_to_yaml_file> --prompt_file <path_to_prompt_file> --infer
```

- `--yaml_path`: Path to the YAML file containing both sets of arguments.
- `--prompt_file`: Path to the text file containing the prompt.
- `--infer`: Perform inference.

Additionally, you can use the `--finetune` flag to perform finetuning.

```bash
python script.py --yaml_path <path_to_yaml_file> --prompt_file <path_to_prompt_file> --finetune
```

## Configuration

### Model Arguments

### Model Configuration

The script supports configuring the model through a YAML file. Below are the available model configuration options:

- `model_name` (optional): Name of the model.
- `cache_dir` (optional): Directory for caching model files.
- `r` (optional): Value for the `r` parameter.
- `lora_alpha` (optional): Value for the `lora_alpha` parameter.
- `lora_dropout` (optional): Value for the `lora_dropout` parameter.
- `bits` (optional): Number of bits.
- `double_quant` (optional): Boolean indicating whether double quantization is used.
- `quant_type`: Type of quantization.
- `trust_remote_code` (optional): Boolean indicating whether to trust remote code.
- `use_auth_token`: Boolean or string indicating whether to use an authentication token.
- `compute_type` (optional): Type of computation.

### Data Training Configuration

For fine-tuning, the script uses data training arguments. Below are the available options:

- `dataset_name` (optional): Name of the dataset.
- `block_size` (optional): Block size for data training.
- `multi_gpu` (optional): Boolean indicating whether to use multiple GPUs.
- `tensor_parallel` (optional): Boolean indicating whether to use tensor parallelism.
- `model_output_dir` (optional): Directory for model output.
- `per_device_train_batch_size` (optional): Batch size per device during training.
- `gradient_accumulation_steps` (optional): Number of gradient accumulation steps.
- `optim` (optional): Optimization algorithm.
- `save_steps` (optional): Number of steps before saving a checkpoint.
- `logging_steps` (optional): Number of steps before logging.
- `learning_rate` (optional): Learning rate for training.
- `max_grad_norm` (optional): Maximum gradient norm for gradient clipping.
- `max_steps` (optional): Maximum number of training steps.
- `warmup_ratio` (optional): Warmup ratio for learning rate schedule.
- `lr_scheduler_type` (optional): Type of learning rate scheduler.
- `compute_type` (optional): Type of computation.

## Examples

### Example 1: Inference

Performing model inference:

```bash
python script.py --yaml_path config.yaml --prompt_file prompt.txt --infer
```

### Example 2: Finetuning

Performing model finetuning:

```bash
python script_name.py --yaml_path path/to/config.yaml --finetune
```

## Error Handling

The script provides error handling for various exceptions. If an error occurs, the script logs the details and exits gracefully.

## Contributing

Contributions to the project are welcome. If you find a bug or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [Apache-2.0 license](LICENSE). Feel free to use and modify the code according to the terms of the license.

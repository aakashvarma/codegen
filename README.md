# codegen

## Introduction

This documentation provides detailed information on the installation, usage, configuration, error handling, contributing guidelines, and license information for the script designed for model inference and finetuning. The script is tailored to work with a specific model and offers customizable behavior through YAML configuration files.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Fine-tuning](#fine-tuning)
  - [Model Merging](#model-merging)
  - [Inference](#inference)
  - [Validation](#validation)
- [Configuration](#configuration)
  - [Example YAML Configuration](#example-yaml-configuration)
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
Certainly! Here are the commands with duplicates removed:

### Usage

Before running the commands, ensure that the necessary configuration files are set up according to the provided examples.

#### Fine-tuning

```bash
python3 main.py --model_yaml <path-to-model-config-yaml> --trainer_yaml <path-to-trainer-config-yaml> --finetune_yaml <path-to-finetune-config-yaml> --finetune
```

#### Model Merging

```bash
python3 main.py --model_yaml <path-to-model-config-yaml> --model_path <path-to-model> --merge_adapter
```
#### Inference

Inference of the original huggingface model
```bash
python3 main.py --model_yaml <path-to-model-config-yaml> --prompt_file <path-to-prompt-text-file> --infer
```

Inference of the model (or merged model) without LoRA adapters from path
```bash
python3 main.py --model_yaml <path-to-model-config-yaml> --prompt_file <path-to-prompt-text-file> --model_path <path-to-model> --infer
```

Inference of the model with LoRA adapters from path
```bash
python3 main.py --model_yaml <path-to-model-config-yaml> --prompt_file <path-to-prompt-text-file> --model_path <path-to-model> --model_with_adapter --infer
```

#### Validation
--validation_dir <path-to-validation-data-directory> should be passed for model validation additionally. <path-to-validation-data-directory> should be a pickle file. 

Validation of the model (or merged model) without LoRA adapters from path
```bash
python3 main.py --model_yaml <path-to-model-config-yaml> --validation_dir <path-to-validation-data-directory> --model_path <path-to-model> --validate
```

Validation of the model with LoRA adapters from path
```bash
python3 main.py --model_yaml <path-to-model-config-yaml> --validation_dir <path-to-validation-data-directory> --model_path <path-to-model> --model_with_adapter --validate
```

Replace `<path-to-...>` with the appropriate paths to your configuration files, data files, and model checkpoints. Additionally, ensure that the appropriate CUDA visible devices are set if using GPU acceleration (`CUDA_VISIBLE_DEVICES`).

### Example YAML Configuration

Below is an example of how the YAML configuration files should be structured:

#### Model Configuration YAML:

- `model_name`: The name or identifier of the pre-trained model to be used. For example, `"codellama/CodeLlama-7b-hf"`.
- `cache_dir`: Directory path where the pre-trained model cache will be stored. If not specified, it defaults to `None`.
- `r`: An integer representing a hyperparameter (`r`) used in the model. Its specific significance would be defined by the model architecture or implementation.
- `lora_alpha`: Another hyperparameter (`lora_alpha`) used in the model, typically specific to the model's architecture or design.
- `lora_dropout`: Dropout rate for regularization in the model. Dropout is a technique used to prevent overfitting in neural networks.
- `bits`: Number of bits used for quantization. Quantization is a technique used to reduce the memory and computational requirements of deep learning models.
- `double_quant`: Boolean indicating whether double quantization is used. Double quantization is a technique that quantizes weights and activations separately.
- `quant_type`: Type of quantization method used, such as `"nf4"` (which specific type is not clear without further context).
- `trust_remote_code`: Boolean indicating whether to trust remote code sources. This could be relevant when loading models or dependencies from external sources.
- `use_auth_token`: Boolean indicating whether to use an authentication token. This might be necessary for accessing certain models or resources.
- `compute_type`: Type of computation used, such as `"fp16"` for 16-bit floating point arithmetic.

```yaml
model_config:
    model_name: "codellama/CodeLlama-7b-hf"
    cache_dir: null
    r: 64
    lora_alpha: 32.0
    lora_dropout: 0.05
    bits: 4
    double_quant: true
    quant_type: "nf4"
    trust_remote_code: false
    use_auth_token: false
    compute_type: "fp16"
```

#### Trainer Configuration YAML:

- `dataset_name`: Name or identifier of the dataset used for training.
- `block_size`: Maximum length of input sequences (usually in tokens).
- `multi_gpu`: Boolean indicating whether to use multiple GPUs for training.
- `tensor_parallel`: Boolean indicating whether to use tensor parallelism for training.
- `model_output_dir`: Directory where the trained model will be saved.
- `per_device_train_batch_size`: Batch size per GPU for training.
- `gradient_accumulation_steps`: Number of steps before gradient accumulation. Useful for training with large batch sizes.
- `optim`: Optimization algorithm used for training (e.g., `"paged_adamw_32bit"`).
- `save_steps`: Frequency of saving checkpoints during training.
- `logging_steps`: Frequency of logging training statistics.
- `learning_rate`: Initial learning rate for training.
- `max_grad_norm`: Maximum gradient norm, used for gradient clipping.
- `max_steps`: Maximum number of training steps.
- `warmup_ratio`: Ratio of warmup steps to total training steps.
- `lr_scheduler_type`: Type of learning rate scheduler used during training.
- `compute_type`: Type of computation used, typically defined in terms of precision (e.g., `"fp16"` for 16-bit floating point arithmetic).
- `num_train_epochs`: Number of training epochs.
- `evaluation_strategy`: Strategy for evaluating the model during training, such as `"steps"` or `"epoch"`.

```yaml
trainer_config:
    dataset_name: "b-mc2/sql-create-context"
    block_size: 512
    multi_gpu: false
    tensor_parallel: false
    model_output_dir: "__run.default"
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4
    optim: "paged_adamw_32bit"
    save_steps: 100
    logging_steps: 10
    learning_rate: 0.0002
    max_grad_norm: 0.3
    max_steps: 100
    warmup_ratio: 0.03
    lr_scheduler_type: "constant"
    compute_type: "fp16"
    num_train_epochs: 1
    evaluation_strategy: "steps"
```

#### Fine-tune Configuration YAML:

- `r`: Hyperparameter specific to the fine-tuning process, with its meaning likely related to the fine-tuning algorithm or model architecture.
- `lora_alpha`: Another hyperparameter used in fine-tuning, possibly related to regularization or optimization.
- `lora_dropout`: Dropout rate specific to fine-tuning, similar to the `lora_dropout` in the model configuration.

```yaml
finetune_config:
    r: 16
    lora_alpha: 32.0
    lora_dropout: 0.05
```


## Error Handling

The script provides comprehensive error handling for various exceptions. In case of an error, the script logs details and exits gracefully.

## Contributing

Contributions to the project are welcomed. If you encounter a bug or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [Apache-2.0 license](LICENSE). Feel free to use and modify the code according to the terms of the license.
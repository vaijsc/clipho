#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language. Currently this script supports the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=fill-mask)
"""

import logging
import os
import sys
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Optional
from PIL import PngImagePlugin, Image

import torch
from torch.utils.data import DataLoader, SequentialSampler
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    VisionTextDualEncoderConfig,
    AutoImageProcessor,
    PreTrainedModel,
    VisionTextDualEncoderModel,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, seed_worker
from transformers.utils import check_min_version, send_example_telemetry, is_datasets_available
from transformers.utils.versions import require_version
from transformers.models.clip.modeling_clip import CLIPOutput


LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
Image.MAX_IMAGE_PIXELS = None

SEED = 2024
random.seed(SEED)

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.46.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim_args: Optional[Dict[str, int]] = field(default_factory=dict)
    optim_target_modules: Optional[List[str]] = field(default_factory=lambda: ["*"])


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    use_vision_text_dual_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to use CustomVisionTextDualEncoderModel class"},
    )
    use_fast_processor: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_flash_attn_2: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2"}
    )
    enable_gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Manually enable gradient checkpoiting"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    unfreeze_vision_modules: Optional[List[str]] = field(
        default=None, 
        metadata={"help": "List of specific vision model submodule names (dot-separated relative to vision_model) to keep unfrozen, even if freeze_vision_model is True. Example: 'encoder.layers.10' 'encoder.layers.11.mlp'"}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )
    unfreeze_text_modules: Optional[List[str]] = field(
        default=None, 
        metadata={"help": "List of specific text model submodule names (dot-separated relative to text_model) to keep unfrozen, even if freeze_text_model is True. Example: 'encoder.layer.11' 'pooler'"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    image_data_paths: str = field(
        default=None, 
        metadata={"help": "A list of image dataset paths"}
    )
    image_column: Optional[str] = field(
        default="image",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_data_paths: str = field(
        default=None, 
        metadata={"help": "A list of caption files (.jsonl files)"}
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    selected_indices_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to indices file to select from image_caption_dataset (a json file)."},
    )
    removed_indices_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to indices file to remove from image_caption_dataset (a json file)."},
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    custom_order: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to put data samples specify in selected_indices_file at the end of dataset"},
    )
    shuffle_data: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to shuffle datasets. "
                "When using this arg, the entire dataset will be shuffled if custom_order=False else each part will be shuffled seperately"
            )
        }
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x
    

class ImageCaptionDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_dataset=None, caption_dataset=None):
        self.image_dataset = image_dataset
        self.caption_dataset = caption_dataset

    def __len__(self) -> int:
        return len(self.image_dataset)
    
    def __getitem__(self, index: int):
        if "attention_mask" in self.caption_dataset[index].keys():
            return {
                "pixel_values": self.image_dataset[index]["pixel_values"],
                "input_ids": self.caption_dataset[index]["input_ids"],
                "attention_mask": self.caption_dataset[index]["attention_mask"],
            }
        else:
            return {
                "pixel_values": self.image_dataset[index]["pixel_values"],
                "input_ids": self.caption_dataset[index]["input_ids"],
            }
    
    def load_from_disk(self, image_data_paths: str, caption_data_paths: str, 
        image_column: str="image", caption_column: str="caption", 
        selected_indices_file: str=None, removed_indices_file: str=None,
        custom_order: bool=True, shuffle: bool=True
        ):
        image_dataset = []
        for file in image_data_paths:
            try:
                image_data_subset = load_dataset(file)
                if image_data_subset["train"] and image_data_subset["validation"]:
                    image_data_subset = concatenate_datasets([image_data_subset["train"], image_data_subset["validation"]])
                image_data_subset = image_data_subset.rename_column("jpg", image_column)
            except:
                image_data_subset = load_from_disk(file)
            image_data_subset = image_data_subset.remove_columns([column for column in image_data_subset.column_names if column != image_column])
            image_dataset.append(image_data_subset)
        image_dataset = concatenate_datasets(image_dataset)

        caption_dataset = []
        for file in caption_data_paths:
            caption_data_subset = load_dataset("json", data_files=file)["train"]
            caption_data_subset = caption_data_subset.remove_columns([column for column in caption_data_subset.column_names if column != caption_column])
            caption_dataset.append(caption_data_subset)
        caption_dataset = concatenate_datasets(caption_dataset)

        if selected_indices_file:
            with open(selected_indices_file) as f:
                selected_indices = json.load(f)
        else:
            selected_indices = list(range(len(image_dataset)))

        if removed_indices_file:
            with open(removed_indices_file) as f:
                removed_indices = json.load(f)
            selected_indices = sorted(list(set(selected_indices) - set(removed_indices)))

        if not shuffle:
            if custom_order:
                if not selected_indices_file:
                    raise Exception("Custom order needs selected_indices_file to put to the end of dataset")
                else:
                    remaining_indices = set(list(range(len(image_dataset)))) - set(selected_indices)
                    if removed_indices_file:
                        remaining_indices = remaining_indices - set(removed_indices)
                    remaining_indices = sorted(list(remaining_indices))

                    selected_image_dataset = image_dataset.select(selected_indices)
                    remaining_image_dataset = image_dataset.select(remaining_indices)
                    selected_caption_dataset = caption_dataset.select(selected_indices)
                    remaining_caption_dataset = caption_dataset.select(remaining_indices)

                    image_dataset = concatenate_datasets([remaining_image_dataset, selected_image_dataset])
                    caption_dataset = concatenate_datasets([remaining_caption_dataset, selected_caption_dataset])
            else:
                image_dataset = image_dataset.select(selected_indices)
                caption_dataset = caption_dataset.select(selected_indices)
        else:
            if custom_order:
                if not selected_indices_file:
                    raise Exception("Custom order needs selected_indices_file to put to the end of dataset")
                else:
                    remaining_indices = set(list(range(len(image_dataset)))) - set(selected_indices)
                    if removed_indices_file:
                        remaining_indices = remaining_indices - set(removed_indices)
                    remaining_indices = list(remaining_indices)

                    # Shuffle each part of the dataset
                    random.shuffle(selected_indices)
                    random.shuffle(remaining_indices)

                    selected_image_dataset = image_dataset.select(selected_indices)
                    remaining_image_dataset = image_dataset.select(remaining_indices)
                    selected_caption_dataset = caption_dataset.select(selected_indices)
                    remaining_caption_dataset = caption_dataset.select(remaining_indices)

                    image_dataset = concatenate_datasets([remaining_image_dataset, selected_image_dataset])
                    caption_dataset = concatenate_datasets([remaining_caption_dataset, selected_caption_dataset])
            else:
                # Shuffle the entire dataset
                random.shuffle(selected_indices)
                image_dataset = image_dataset.select(selected_indices)
                caption_dataset = caption_dataset.select(selected_indices)

        self.image_dataset = image_dataset
        self.caption_dataset = caption_dataset
    

class CustomVisionTextDualEncoderModel(VisionTextDualEncoderModel):

    def __init__(
        self,
        config: Optional[VisionTextDualEncoderConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config, vision_model, text_model)
        self.logit_scale = torch.nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        if self.config.loss_type == "siglip_style":
            self.logit_bias = torch.nn.Parameter(torch.tensor(self.config.logit_bias_init_value))

    def compute_loss(self, logits_per_text):
        """
        Overridden loss computation method to support both clip_style and siglip_style losses.
        
        Args:
            logits_per_text (torch.Tensor): logits_per_text of a batch.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

        if self.config.loss_type == "clip_style":
            caption_loss = contrastive_loss(logits_per_text)
            image_loss = contrastive_loss(logits_per_text.t())
            loss = (caption_loss + image_loss) / 2.0
            return loss
        elif self.config.loss_type == "siglip_style":
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()
            return loss
        else:
            raise ValueError(f"Unsupported loss_type: {self.config.loss_type}. Choose 'clip_style' or 'siglip_style'.")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CLIPOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]  # pooler_output
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]  # pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        if self.config.loss_type == "clip_style":
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale.exp()
            logits_per_image = logits_per_text.t()
        else:
            logits_per_text = (
                torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * self.logit_scale.exp()
                + self.logit_bias
            )
            logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = self.compute_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)

    attention_mask = []
    for example in examples:
        if "attention_mask" in example.keys():
            attention_mask.append(example["attention_mask"])
    if attention_mask != []:
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": True,
        }
    else:
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "return_loss": True,
        }

def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False

# Function to handle selective freezing/unfreezing
def apply_freezing_strategy(module, module_prefix: str, freeze_all: bool, unfreeze_patterns: Optional[List[str]]):
    """
    Applies freezing/unfreezing to a module based on arguments.

    Args:
        module: The PyTorch module (e.g., model.vision_model).
        module_prefix: The name prefix for parameters within this module (e.g., "vision_model").
        freeze_all: Boolean flag indicating if the entire module should generally be frozen.
        unfreeze_patterns: List of submodule name patterns to keep unfrozen if freeze_all is True.
    """
    if not hasattr(module, 'named_parameters'):
        logger.warning(f"Module '{module_prefix}' does not have named_parameters. Skipping freezing logic.")
        return
        
    total_params = 0
    trainable_params = 0

    # Default: set requires_grad based on the freeze_all flag
    initial_grad_state = not freeze_all
    for param in module.parameters():
        param.requires_grad = initial_grad_state

    # If freezing is enabled, selectively unfreeze specific modules
    if freeze_all and unfreeze_patterns:
        logger.info(f"Attempting to unfreeze patterns within {module_prefix}: {unfreeze_patterns}")
        unfrozen_by_pattern = set() # Keep track of params unfrozen by patterns

        # Iterate through all named parameters to find matches
        for name, param in module.named_parameters():
            # The name is relative to the module passed in, e.g., "encoder.layers.10.mlp.fc1.weight"
            # Check if this parameter's module name starts with any of the unfreeze patterns
            should_unfreeze = False
            for pattern in unfreeze_patterns:
                # Match if name is exactly the pattern or starts with pattern + '.'
                if name == pattern or name.startswith(pattern + '.'):
                    should_unfreeze = True
                    break
            
            if should_unfreeze:
                if not param.requires_grad: # Only log/change if it was previously frozen
                    # logger.debug(f"Unfreezing parameter '{module_prefix}.{name}' due to pattern match.")
                    param.requires_grad = True
                    unfrozen_by_pattern.add(f"{module_prefix}.{name}") 
                # else: # If it was already True (e.g. freeze_all=False), no change needed

        if unfrozen_by_pattern:
            logger.info(f"Successfully unfroze {len(unfrozen_by_pattern)} parameter groups in {module_prefix} based on patterns.")
            # You could log the specific names if needed for debugging: logger.info(f"Unfrozen parameters: {unfrozen_by_pattern}")
        else:
            logger.warning(f"No parameters matched the unfreeze patterns {unfreeze_patterns} within {module_prefix}. Check pattern names.")


    # Log final status
    for name, param in module.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params += 1
             
    logger.info(
        f"Module '{module_prefix}': Status - {'FROZEN' if freeze_all else 'TRAINABLE'}. "
        f"Trainable parameters: {trainable_params}/{total_params}."
    )
    if freeze_all and not unfreeze_patterns and trainable_params > 0:
        logger.warning(f"Module '{module_prefix}' was intended to be fully frozen, but {trainable_params} params remain trainable. Check logic.")
    if not freeze_all and trainable_params != total_params:
        logger.warning(f"Module '{module_prefix}' was intended to be fully trainable, but {total_params - trainable_params} params are frozen. Check logic.")


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clip", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    image_column = data_args.image_column
    caption_column = data_args.caption_column

    print("### Start loading datasets")
    image_caption_dataset = ImageCaptionDataset()
    image_caption_dataset.load_from_disk(
        image_data_paths=data_args.image_data_paths.split(','), 
        caption_data_paths=data_args.caption_data_paths.split(','), 
        image_column=image_column, 
        caption_column=caption_column,
        selected_indices_file=data_args.selected_indices_file,
        removed_indices_file=data_args.removed_indices_file,
        custom_order=data_args.custom_order,
        shuffle=data_args.shuffle_data
    )
    print("Done loading datasets")
    image_dataset, caption_dataset = image_caption_dataset.image_dataset, image_caption_dataset.caption_dataset

    if training_args.do_predict:
        val_test_indices = sorted(random.sample(list(range(len(image_dataset))), k=int(len(image_dataset)*0.01)))
        train_indices = sorted(set(list(range(len(image_dataset)))) - set(val_test_indices))
        val_indices = sorted(random.sample(val_test_indices, k=int(len(val_test_indices))*0.5)) 
        test_indices = sorted(set(val_test_indices) - set(val_indices))
        
        train_image_dataset = image_dataset.select(train_indices)
        eval_image_dataset = image_dataset.select(val_indices)
        test_image_dataset = image_dataset.select(test_indices)

        train_caption_dataset = caption_dataset.select(train_indices)
        eval_caption_dataset = caption_dataset.select(val_indices)
        test_caption_dataset = caption_dataset.select(test_indices)
    elif training_args.do_eval:
        val_indices = sorted(random.sample(list(range(len(image_dataset))), k=int(len(image_dataset)*0.005)))
        train_indices = sorted(set(list(range(len(image_dataset)))) - set(val_indices))
        
        train_image_dataset = image_dataset.select(train_indices)
        eval_image_dataset = image_dataset.select(val_indices)

        train_caption_dataset = caption_dataset.select(train_indices)
        eval_caption_dataset = caption_dataset.select(val_indices)
    else:
        train_image_dataset = image_dataset
        train_caption_dataset = caption_dataset
    
    print("Image datasets")
    print(image_dataset)
    print("Caption datasets")
    print(caption_dataset)

    # 5. Load pretrained model, tokenizer, and image processor
    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_processor,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_processor,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model_args.use_vision_text_dual_encoder:
        if model_args.use_flash_attn_2:
            model = CustomVisionTextDualEncoderModel.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
        else:
            model = CustomVisionTextDualEncoderModel.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        if model_args.use_flash_attn_2:
            model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )

    # Manually enable gradient checkpointing
    if model_args.enable_gradient_checkpointing:
        if model_args.use_vision_text_dual_encoder:
            model.text_model.gradient_checkpointing_enable()
            model.vision_model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()

    config = model.config
    if model_args.use_vision_text_dual_encoder:
        model.logit_scale = torch.nn.Parameter(torch.tensor(config.logit_scale_init_value))

    # Apply freeze & unfreeze strategy
    # if model_args.freeze_vision_model:
    #     freeze_params(model.vision_model)
    if hasattr(model, 'vision_model'):
        logger.info(f"Applying freezing strategy to vision_model...")
        apply_freezing_strategy(
            module=model.vision_model,
            module_prefix="vision_model", # Adjust if the attribute name is different
            freeze_all=model_args.freeze_vision_model,
            unfreeze_patterns=model_args.unfreeze_vision_modules
        )
    else:
        logger.warning("Model does not have a 'vision_model' attribute. Skipping vision model freezing.")

    # if model_args.freeze_text_model:
    #     freeze_params(model.text_model)
    if hasattr(model, 'text_model'):
        logger.info(f"Applying freezing strategy to text_model...")
        apply_freezing_strategy(
            module=model.text_model,
            module_prefix="text_model", # Adjust if the attribute name is different
            freeze_all=model_args.freeze_text_model,
            unfreeze_patterns=model_args.unfreeze_text_modules
        )
    else:
        logger.warning("Model does not have a 'text_model' attribute. Skipping text model freezing.")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    # 6. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    # image_transformations = Transform(
    #     config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
    # )
    # image_transformations = torch.jit.script(image_transformations)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        if "attention_mask" in text_inputs.keys():
            examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        # images = [torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).to(torch.uint8) for pil_image in examples[image_column]]
        # examples["pixel_values"] = [image_transformations(image) for image in images]
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = image_processor(images, return_tensors="pt").pixel_values
        return examples
    
    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_image_dataset), data_args.max_train_samples)
            train_image_dataset = train_image_dataset.select(range(max_train_samples))
            train_caption_dataset = train_caption_dataset.select(range(max_train_samples))

        # Create image caption dataset
        train_dataset = ImageCaptionDataset(train_image_dataset, train_caption_dataset)

        print("### Start tokenizing caption")
        train_dataset.caption_dataset = train_dataset.caption_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        print("### Done tokenizing caption")

        # Transform on the fly as doing it on the whole dataset takes too much time.
        train_dataset.image_dataset.set_transform(transform_images)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_image_dataset), data_args.max_eval_samples)
            eval_image_dataset = eval_image_dataset.select(range(max_eval_samples))
            eval_caption_dataset = eval_caption_dataset.select(range(max_eval_samples))

        eval_dataset = ImageCaptionDataset(eval_image_dataset, eval_caption_dataset)

        eval_dataset.caption_dataset = eval_dataset.caption_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        eval_dataset.image_dataset.set_transform(transform_images)

    if training_args.do_predict:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(test_image_dataset), data_args.max_eval_samples)
            test_image_dataset = test_image_dataset.select(range(max_eval_samples))
            test_caption_dataset = test_caption_dataset.select(range(max_eval_samples))

        test_dataset = ImageCaptionDataset(test_image_dataset, test_caption_dataset)

        test_dataset.caption_dataset = test_dataset.caption_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        test_dataset.image_dataset.set_transform(transform_images)

    # 8. Initialize our trainer
    if data_args.custom_order:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            data_collator=collate_fn,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            data_collator=collate_fn,
        )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        image_processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Write Training Stats and push to hub.
    finetuned_from = model_args.model_name_or_path
    # If from a local directory, don't set `finetuned_from` as this is required to be a valid repo. id on the Hub.
    if os.path.isdir(finetuned_from):
        finetuned_from = None
    kwargs = {"finetuned_from": finetuned_from, "tasks": "contrastive-image-text-modeling"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()

from typing import Callable, Dict, List, Optional, Union

import torch
from transformers import Trainer, Seq2SeqTrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from trl import AutoModelForCausalLMWithValueHead

from src.extras.logging import get_logger
from src.hparams import FinetuningArguments, ModelArguments, DataArguments
from src.model import load_model, load_tokenizer, load_valuehead_params


logger = get_logger(__name__)


class DummyOptimizer(torch.optim.Optimizer):
    r"""
    A dummy optimizer used for the GaLore algorithm.
    """

    def __init__(
        self, lr: float = 1e-3, optimizer_dict: Optional[Dict["torch.nn.Parameter", "torch.optim.Optimizer"]] = None
    ) -> None:
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": lr})

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["llama-factory", finetuning_args.finetuning_type],
    }
    if data_args.dataset is not None:
        kwargs["dataset"] = [dataset.strip() for dataset in data_args.dataset.split(",")]

    if model_args.use_unsloth:
        kwargs["tags"] = kwargs["tags"] + ["unsloth"]

    if not training_args.do_train:
        pass
    elif training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(license="other", **kwargs)  # prevent from connecting to hub


def create_ref_model(
    model_args: "ModelArguments", finetuning_args: "FinetuningArguments", add_valuehead: bool = False
) -> Optional[Union["PreTrainedModel", "AutoModelForCausalLMWithValueHead"]]:
    r"""
    Creates reference model for PPO/DPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.
    """
    if finetuning_args.ref_model is not None:
        ref_model_args_dict = model_args.to_dict()
        ref_model_args_dict.update(
            dict(
                model_name_or_path=finetuning_args.ref_model,
                adapter_name_or_path=finetuning_args.ref_model_adapters,
                quantization_bit=finetuning_args.ref_model_quantization_bit,
            )
        )
        ref_model_args = ModelArguments(**ref_model_args_dict)
        ref_finetuning_args = FinetuningArguments(finetuning_type="lora")
        tokenizer = load_tokenizer(ref_model_args)["tokenizer"]
        ref_model = load_model(
            tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
        )
        logger.info("Created reference model from {}".format(finetuning_args.ref_model))
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            tokenizer = load_tokenizer(model_args)["tokenizer"]
            ref_model = load_model(
                tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            )
            logger.info("Created reference model from the model itself.")

    return ref_model


def create_reward_model(
    model: "AutoModelForCausalLMWithValueHead", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> Optional["AutoModelForCausalLMWithValueHead"]:
    r"""
    Creates reward model for PPO training.
    """
    if finetuning_args.reward_model_type == "api":
        assert finetuning_args.reward_model.startswith("http"), "Please provide full url."
        logger.info("Use reward server {}".format(finetuning_args.reward_model))
        return finetuning_args.reward_model
    elif finetuning_args.reward_model_type == "lora":
        model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")
        for name, param in model.named_parameters():  # https://github.com/huggingface/peft/issues/1090
            if "default" in name:
                param.data = param.data.to(torch.float32)  # trainable params should in fp32
        vhead_params = load_valuehead_params(finetuning_args.reward_model, model_args)
        assert vhead_params is not None, "Reward model is not correctly loaded."
        model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
        model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
        model.register_buffer(
            "default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False
        )
        model.register_buffer(
            "default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False
        )
        logger.info("Loaded adapter weights of reward model from {}".format(finetuning_args.reward_model))
        return None
    else:
        reward_model_args_dict = model_args.to_dict()
        reward_model_args_dict.update(
            dict(
                model_name_or_path=finetuning_args.reward_model,
                adapter_name_or_path=finetuning_args.reward_model_adapters,
                quantization_bit=finetuning_args.reward_model_quantization_bit,
            )
        )
        reward_model_args = ModelArguments(**reward_model_args_dict)
        reward_finetuning_args = FinetuningArguments(finetuning_type="lora")
        tokenizer = load_tokenizer(reward_model_args)["tokenizer"]
        reward_model = load_model(
            tokenizer, reward_model_args, reward_finetuning_args, is_trainable=False, add_valuehead=True
        )
        logger.info("Loaded full weights of reward model from {}".format(finetuning_args.reward_model))
        logger.warning("Please ensure the ppo model and reward model share SAME tokenizer and vocabulary.")
        return reward_model


def _get_decay_parameter_names(model: "PreTrainedModel") -> List[str]:
    r"""
    Returns a list of names of parameters with weight decay. (weights in non-layernorm layers)
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def _create_loraplus_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    default_lr = training_args.learning_rate
    loraplus_lr = training_args.learning_rate * finetuning_args.loraplus_lr_ratio
    embedding_lr = finetuning_args.loraplus_lr_embedding

    decay_param_names = _get_decay_parameter_names(model)
    param_dict: Dict[str, List["torch.nn.Parameter"]] = {
        "lora_a": [],
        "lora_b": [],
        "lora_b_nodecay": [],
        "embedding": [],
    }
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_embedding_B" in name:
                param_dict["embedding"].append(param)
            elif "lora_B" in name or param.ndim == 1:
                if name in decay_param_names:
                    param_dict["lora_b"].append(param)
                else:
                    param_dict["lora_b_nodecay"].append(param)
            else:
                param_dict["lora_a"].append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=param_dict["lora_a"], lr=default_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b"], lr=loraplus_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b_nodecay"], lr=loraplus_lr, weight_decay=0.0),
        dict(params=param_dict["embedding"], lr=embedding_lr, weight_decay=training_args.weight_decay),
    ]
    optimizer = optim_class(param_groups, **optim_kwargs)
    logger.info("Using LoRA+ optimizer with loraplus lr ratio {:.2f}.".format(finetuning_args.loraplus_lr_ratio))
    return optimizer


def create_custom_optimzer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Optional["torch.optim.Optimizer"]:

    if finetuning_args.loraplus_lr_ratio is not None:
        return _create_loraplus_optimizer(model, training_args, finetuning_args)


def create_custom_scheduler(
    training_args: "Seq2SeqTrainingArguments",
    num_training_steps: int,
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> None:
    if optimizer is not None and isinstance(optimizer, DummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict: Dict["torch.nn.Parameter", "torch.optim.lr_scheduler.LRScheduler"] = {}

        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer=optimizer_dict[param],
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

        def scheduler_hook(param: "torch.nn.Parameter"):
            scheduler_dict[param].step()

        for param in optimizer_dict.keys():
            param.register_post_accumulate_grad_hook(scheduler_hook)

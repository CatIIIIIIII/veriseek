# Inspired by: https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py

import re
import pickle
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, TrainerCallback
from ...data import get_dataset
from ...extras.callbacks import FixValueHeadModelCallback
from ...extras.misc import fix_valuehead_checkpoint
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_ref_model
from .trainer import CustomPPOCompilerTrainer


from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

def extract_module_name(text):
    # Define the regex pattern to match the module name between 'Module name:' and 'Input ports:'
    pattern = r'Module name:\n*\s*(\w+)\s*\n'
    
    # Search for the pattern in the provided text
    match = re.search(pattern, text)
    
    assert match, f"No module name found.{text}"
    return match.group(1).strip()
        

def run_ppo_compiler(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="ppo_compiler", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    # load pickle file opencores_tree.pkl
    with open("instruct_opencores_ast.pkl", "rb") as f:
        golden_code_tree = pickle.load(f)    
    
    golden_code = {}
    for data in dataset:
        prompt = tokenizer.decode(data["input_ids"], skip_special_tokens=True)
        code = tokenizer.decode(data["labels"], skip_special_tokens=True)
        module_name = extract_module_name(prompt)
        golden_code[module_name] = code

    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)

    # Initialize our Trainer
    ppo_trainer = CustomPPOCompilerTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks + [FixValueHeadModelCallback()],
        model=model,
        ref_model=ref_model,
        dataset=dataset,
        data_collator=data_collator,
        golden_code=golden_code,
        golden_code_tree=golden_code_tree,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        ppo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        ppo_trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)
        ppo_trainer.save_state()  # must be called after save_model to have a folder
        if ppo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
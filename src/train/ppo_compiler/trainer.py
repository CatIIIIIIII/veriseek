import math
import re
import os
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
import pyverilog.vparser.ast as vast
from pyverilog.vparser.parser import parse

from ...extras.callbacks import FixValueHeadModelCallback, LogCallback
from ...extras.logging import get_logger
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..utils import create_custom_optimzer, create_custom_scheduler
from .utils import dump_layernorm, restore_layernorm


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)



class ParseTreeNormalizer:
    def normalize(self, node):
        if isinstance(node, vast.Node):
            normalized_children = [self.normalize(c) for c in node.children()]
            if isinstance(node, vast.Identifier):
                return ('Identifier',)  # Ignore the actual name
            if isinstance(node, vast.IntConst):
                return ('IntConst',)  # Ignore the actual value
            return (node.__class__.__name__, tuple(normalized_children))
        elif isinstance(node, list):
            return [self.normalize(n) for n in node]
        return node

# def compare_trees(tree1, tree2):
#     if tree1 == tree2:
#         return 1.0  # Trees are identical
#     if isinstance(tree1, tuple) and isinstance(tree2, tuple):
#         if tree1[0] == tree2[0]:
#             children1 = tree1[1]
#             children2 = tree2[1]
#             if len(children1) == len(children2):
#                 return sum(compare_trees(c1, c2) for c1, c2 in zip(children1, children2)) / len(children1)
#             else:
#                 # Penalize for different number of children
#                 return sum(compare_trees(c1, c2) for c1, c2 in zip(children1, children2)) / max(len(children1), len(children2))
#     return 0.0  # Trees are different


def compare_trees(tree1, tree2):
    if tree1 == tree2:
        return 1.0

    if isinstance(tree1, tuple) and isinstance(tree2, tuple) and tree1[0] == tree2[0]:
        children1 = set(tree1[1:]) if len(tree1) > 1 else set()
        children2 = set(tree2[1:]) if len(tree2) > 1 else set()
        
        s = 0
        compared_children = set()
        for child1 in children1:
            best_match = 0
            best_child = None
            for child2 in children2:
                if child2 not in compared_children:
                    if isinstance(child1, tuple) and isinstance(child2, tuple) and child1[0] == child2[0]:
                        similarity = sim_ast(child1, child2)
                        if similarity > best_match:
                            best_match = similarity
                            best_child = child2
            if best_child:
                s += best_match
                compared_children.add(best_child)
        
        max_count = max(len(children1), len(children2))
        if max_count > 0:
            return s / max_count
        else:
            return 1.0

    return 0.0


def extract_module_name(text):
    # Define the regex pattern to match the module name between 'Module name:' and 'Input ports:'
    pattern = r'Module name:\n*\s*(\w+)\s*\n'
    
    # Search for the pattern in the provided text
    match = re.search(pattern, text)
    
    assert match, f"No module name found.{text}"
    return match.group(1).strip()

class CustomPPOCompilerTrainer(PPOTrainer, Trainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: List["TrainerCallback"],
        model: "AutoModelForCausalLMWithValueHead",
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        dataset: "Dataset",
        data_collator: "DataCollatorWithPadding",
        golden_code: Dict,
        golden_code_tree: Dict,
    ):
        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(len(dataset) / total_train_batch_size)

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.current_device = get_current_device()  # patch for deepspeed training
        self.processor = processor

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        assert isinstance(self.log_callback, LogCallback) and isinstance(self.save_callback, FixValueHeadModelCallback)

        if self.args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        self.golden_code = golden_code
        self.golden_code_tree = {k.lower(): v for k, v in golden_code_tree.items()}

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info("  Num examples = {}".format(num_examples))
            logger.info("  Num Epochs = {}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {}".format(self.args.per_device_train_batch_size))
            logger.info(
                "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {}".format(
                    total_train_batch_size
                )
            )
            logger.info("  Gradient Accumulation steps = {}".format(self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {}".format(self.finetuning_args.ppo_epochs))
            logger.info("  Total training steps = {}".format(max_steps))
            logger.info("  Number of trainable parameters = {}".format(count_parameters(self.model)[0]))

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Cast to inference mode
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            self.model.eval()

            # Get inputs
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch_queries, mini_batch_responses = self.get_inputs(
                    batch[idx : idx + self.config.mini_batch_size]
                )
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            # Cast to training mode
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False
            self.model.train()

            # Run PPO step
            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.log_callback.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.log_callback.on_log(self.args, self.state, self.control)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step))
                )
                self.save_callback.on_save(
                    self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
                )

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.log_callback.on_train_end(self.args, self.state, self.control)
        self.save_callback.on_train_end(
            self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
        )

    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimzer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        if self.processor is not None:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""
        Generates model's responses given queries.
        """
        if self.model_args.upcast_layernorm:
            layernorm_params = dump_layernorm(self.model)

        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        generate_output: torch.Tensor = unwrapped_model.generate(
            generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
        )

        if self.model_args.upcast_layernorm:
            restore_layernorm(self.model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_index = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_index) == 0:
                response_length = 1  # allow empty response
            else:
                response_length = response_index[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses
    
    @torch.no_grad()
    def get_rewards(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        r"""
        Computes scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        
        responses_text, labels_text, label_ast = self.decode(queries, responses)
        rewards=[]
        for response, label, ast in zip(responses_text, labels_text, label_ast):
            if len(set(response)) < 10:
                reward = -10
            elif "module" in response and "endmodule" in response:
                try:
                    ast_response, _ = parse([response])
                    # Normalize trees
                    normalizer = ParseTreeNormalizer()
                    normalized_response = normalizer.normalize(ast_response)
                    normalized_label = normalizer.normalize(ast)
                    reward = compare_trees(normalized_response, normalized_label) * 10
                    print(f"reward: {reward}")
                    # exit()
                except:
                    reward = 0
            else:
                reward = -5
                
            rewards.append(torch.tensor(reward, dtype=torch.float32))
        return rewards

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with torch.cuda.amp.autocast(dtype=self.model_args.compute_dtype):  # support bf16
                logits, _, values = model(**input_kwargs)

            unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
            if getattr(unwrapped_model.config, "model_type", None) == "chatglm":
                values = torch.transpose(values, 0, 1)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            try:
                self._save(output_dir, state_dict=self.accelerator.get_state_dict(self.model))
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                self._save(output_dir, state_dict={})
                remove_dummy_checkpoint(True, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)
    
    def decode(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
    ):  
        def _stop_at_stop_token(decoded_string, stop_tokens=['<|EOT|>', 'endmodule']):
            min_stop_index = len(decoded_string)
            for stop_token in stop_tokens:
                stop_index = decoded_string.find(stop_token)
                if stop_index != -1 and stop_index < min_stop_index:
                    min_stop_index = stop_index
            return decoded_string[:min_stop_index]
        
        # def filter(gen):
        #     gen = gen.strip('\n')
        #     # remove the wrong comments line start with '#'
        #     gen_lines = gen.split('\n')
        #     gen_lines = [line for line in gen_lines if not line.strip().startswith('#')]
            
        #     open_comment = False
        #     result = []
        #     for line in gen_lines:
        #         stripped_line = line.strip()
        #         if stripped_line.startswith('/*') and stripped_line.endswith('*/'):
        #             continue  # Ignore complete comments
        #         elif stripped_line.startswith('/*'):
        #             if open_comment:
        #                 continue  # Ignore if there's already an open comment
        #             open_comment = True
        #         elif stripped_line.endswith('*/'):
        #             if not open_comment:
        #                 continue  # Ignore unpaired closing comment
        #             open_comment = False
        #         elif '/*' in stripped_line and '*/' not in stripped_line:
        #             open_comment = True
        #         elif '*/' in stripped_line and '/*' not in stripped_line:
        #             if not open_comment:
        #                 continue  # Ignore unpaired closing comment
        #             open_comment = False

        #         result.append(line)

        #     result = '\n'.join(result)
        #     if 'endmodule' not in result and set(result) != {'\n'}:
        #         result += '\nendmodule'
        #     return result
        
        def filter(gen):
            # remove the wrong comments line start with '#'
            gen_lines = gen.split('\n')
            gen_lines = [line for line in gen_lines if not line.strip().startswith('#')]
            return '\n'.join(gen_lines) + "endmodule"
        
        # def filter(gen):
        #     gen = gen.strip('\n')
        #     gen_lines = gen.split('\n')
            
        #     open_comment = False
        #     result = []
        #     for line in gen_lines:
        #         stripped_line = line.strip()
                
        #         # Check for endmodule
        #         if 'endmodule' in stripped_line:
        #             result.append(line)
        #             break  # Stop processing at first endmodule
                
        #         # Skip single-line comments
        #         if stripped_line.startswith('//'):
        #             continue
                
        #         # Handle multi-line comments
        #         if '/*' in stripped_line and '*/' in stripped_line:
        #             continue  # Skip single-line multi-line comments
        #         elif '/*' in stripped_line:
        #             open_comment = True
        #             continue
        #         elif '*/' in stripped_line:
        #             open_comment = False
        #             continue
                
        #         # Skip lines inside multi-line comments
        #         if open_comment:
        #             continue
                
        #         result.append(line)

        #     result = '\n'.join(result)
            
        #     # Add endmodule if not present and the result is not empty
        #     if 'endmodule' not in result and set(result.strip()) != set():
        #         result += '\nendmodule'
            
        #     return result
        
        queries_text = self.tokenizer.batch_decode(
            queries, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        responses_text = self.tokenizer.batch_decode(
            responses, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        responses_text = [_stop_at_stop_token(response) for response in responses_text]
        responses_text = [filter(response) for response in responses_text]

        module_name = [extract_module_name(query) for query in queries_text]
        labels_text = [self.golden_code[name] for name in module_name]
        label_ast = [self.golden_code_tree[name.lower()] for name in module_name]
        
        return responses_text, labels_text, label_ast
        

        
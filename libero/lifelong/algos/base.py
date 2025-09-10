import os
import time

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *

REGISTERED_ALGOS = {}


def register_algo(policy_class):
    """Register a policy class with the registry."""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_ALGOS:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_ALGOS[policy_name] = policy_class


def get_algo_class(algo_name):
    """Get the policy class from the registry."""
    if algo_name.lower() not in REGISTERED_ALGOS:
        raise ValueError(
            "Policy class with name {} not found in registry".format(algo_name)
        )
    return REGISTERED_ALGOS[algo_name.lower()]


def get_algo_list():
    return REGISTERED_ALGOS


class AlgoMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all algorithms that should not be registered here.
        _unregistered_algos = []

        if cls.__name__ not in _unregistered_algos:
            register_algo(cls)
        return cls


class Sequential(nn.Module, metaclass=AlgoMeta):
    """
    The sequential finetuning BC baseline, also the superclass of all lifelong
    learning algorithms.
    """

    def __init__(self, n_tasks, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_scale = cfg.train.loss_scale
        self.n_tasks = n_tasks
        if not hasattr(cfg, "experiment_dir"):
            create_experiment_dir(cfg)
            print(
                f"[info] Experiment directory not specified. Creating a default one: {cfg.experiment_dir}"
            )
        self.experiment_dir = cfg.experiment_dir
        self.algo = cfg.lifelong.algo

        self.policy = get_policy_class(cfg.policy.policy_type)(cfg, cfg.shape_meta)
        self.current_task = -1

    def end_task(self, dataset, task_id, benchmark, env=None):
        """
        What the algorithm does at the end of learning each lifelong task.
        """
        pass

    def start_task(self, task):
        """
        What the algorithm does at the beginning of learning each lifelong task.
        """
        self.current_task = task

        # initialize the optimizer and scheduler
        self.optimizer = eval(self.cfg.train.optimizer.name)(
            self.policy.parameters(), **self.cfg.train.optimizer.kwargs
        )

        self.scheduler = None
        if self.cfg.train.scheduler is not None:
            self.scheduler = eval(self.cfg.train.scheduler.name)(
                self.optimizer,
                T_max=self.cfg.train.n_epochs,
                **self.cfg.train.scheduler.kwargs,
            )

        # if Prompt-Based update process_task_count
        if "Prompt" in self.cfg.policy["policy_type"]:
            self.policy.prompt.process_task_count(task)

        if self.cfg.policy["policy_type"] == "BCHierarchicalTransformerCPPolicy":
            self.policy.task_count = task

    def map_tensor_to_device(self, data):
        """Move data to the device specified by self.cfg.device."""
        return TensorUtils.map_tensor(
            data, lambda x: safe_device(x, device=self.cfg.device)
        )

    def observe(self, data):
        """
        How the algorithm learns on each data point.
        """
        data = self.map_tensor_to_device(data)
        prompt_loss = torch.zeros((1,))
        cp_c_loss = torch.zeros((1,))
        self.optimizer.zero_grad()
        if "WOID" in self.cfg.policy["policy_type"]:
            loss, prompt_loss, cp_c_loss = self.policy.cp_prompt_woid_compute_loss(data)
        elif "Prompt" in self.cfg.policy["policy_type"] and "CP" in self.cfg.policy["policy_type"]:
            loss, prompt_loss = self.policy.cp_prompt_compute_loss(data)
        elif "CP" in self.cfg.policy["policy_type"]:
            loss = self.policy.cp_compute_loss(data)
        elif "Prompt" in self.cfg.policy["policy_type"]:
            loss, prompt_loss = self.policy.prompt_compute_loss(data)
        else:
            loss = self.policy.compute_loss(data)

        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:  # 是否进行梯度裁剪
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip  # 梯度范数的上限，默认为2范数
            )
        self.optimizer.step()
        return loss.item(), prompt_loss.item(), cp_c_loss.item()
        # return loss.item(), prompt_loss.item()

    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        prompt_loss = torch.zeros((1,))
        cp_c_loss = torch.zeros((1,))
        with (torch.no_grad()):
            if "WOID" in self.cfg.policy["policy_type"]:
                loss, prompt_loss, cp_c_loss = self.policy.cp_prompt_woid_compute_loss(data)
            elif "Prompt" in self.cfg.policy["policy_type"] and "CP" in self.cfg.policy["policy_type"]:
                loss, prompt_loss = self.policy.cp_prompt_compute_loss(data)
            elif "CP" in self.cfg.policy["policy_type"]:
                loss = self.policy.cp_compute_loss(data)
            elif "Prompt" in self.cfg.policy["policy_type"]:
                loss, prompt_loss = self.policy.prompt_compute_loss(data)
            else:
                loss = self.policy.compute_loss(data)
        return loss.item(), prompt_loss.item(), cp_c_loss.item()
        # return loss.item(), prompt_loss.item()

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):

        self.start_task(task_id)

        # recover the corresponding manipulation task ids
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_model.pth"
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            persistent_workers=False,
        )

        prev_success_rate = -1.0
        success_rate_drop_time = 0
        best_state_dict = self.policy.state_dict()  # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        bc_losses = []

        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()
            training_loss = 0.0
            training_prompt_loss = 0.0
            training_cp_c_loss = 0.0

            if epoch > 0:  # update
                self.policy.train()
                for (idx, data) in enumerate(train_dataloader):
                    loss, prompt_loss, cp_c_loss = self.observe(data)
                    # loss, prompt_loss = self.observe(data)
                    training_loss += loss
                    training_prompt_loss += prompt_loss
                    training_cp_c_loss += cp_c_loss
                training_loss /= len(train_dataloader)
                training_prompt_loss /= len(train_dataloader)
                training_cp_c_loss /= len(train_dataloader)
            else:  # just evaluate the zero-shot performance on 0-th epoch
                for (idx, data) in enumerate(train_dataloader):
                    loss, prompt_loss, cp_c_loss = self.eval_observe(data)
                    # loss, prompt_loss = self.eval_observe(data)
                    training_loss += loss
                    training_prompt_loss += prompt_loss
                    training_cp_c_loss += cp_c_loss
                training_loss /= len(train_dataloader)
                training_prompt_loss /= len(train_dataloader)
                training_cp_c_loss /= len(train_dataloader)
            t1 = time.time()

            if "WOID" in self.cfg.policy["policy_type"]:
                print(
                    f"[info] Epoch: {epoch:3d} | total train loss: {training_loss:5.4f} | "
                    f"prompt_loss: {training_prompt_loss:5.4f} | cp_c_loss: {training_cp_c_loss:5.4f} | "
                    f"time: {(t1 - t0) / 60:4.2f}"
                )
            elif "Prompt" and "CP" in self.cfg.policy["policy_type"]:
                print(
                    f"[info] Epoch: {epoch:3d} | total train loss: {training_loss:5.4f} | "
                    f"prompt_loss: {training_prompt_loss:5.4f} | "
                    f"time: {(t1 - t0) / 60:4.2f}"
                )
            elif "Prompt" in self.cfg.policy["policy_type"]:
                print(
                    f"[info] Epoch: {epoch:3d} | total train loss: {training_loss:5.4f} | "
                    f"prompt_loss: {training_prompt_loss:5.4f} | "
                    f"time: {(t1 - t0) / 60:4.2f}"
                )
            elif "CP" in self.cfg.policy["policy_type"]:
                print(
                    f"[info] Epoch: {epoch:3d} | bc train loss: {training_loss:5.4f} | "
                    f"time: {(t1 - t0) / 60:4.2f}"
                )
            else:
                print(
                    f"[info] Epoch: {epoch:3d} | bc train loss: {training_loss:5.4f} | "
                    f"time: {(t1 - t0) / 60:4.2f}"
                )

            if epoch % self.cfg.eval.eval_every == 0:  # evaluate BC loss
                # every eval_every epoch, we evaluate the agent on the current task,
                # then we pick the best performant agent on the current task as
                # if it stops learning after that specific epoch. So the stopping
                # criterion for learning a new task is achieving the peak performance
                # on the new task. Future work can explore how to decide this stopping
                # epoch by also considering the agent's performance on old tasks.

                bc_losses.append(training_loss - training_prompt_loss - training_cp_c_loss)
                # bc_losses.append(training_loss - training_prompt_loss)

                t0 = time.time()

                task_str = f"k{task_id}_e{epoch//self.cfg.eval.eval_every}"  # k表示当前正在训练的任务，e表示当前训练过程中的第几次测试
                sim_states = (
                    result_summary[task_str] if self.cfg.eval.save_sim_states else None
                )
                success_rate = evaluate_one_task_success(
                    cfg=self.cfg,
                    algo=self,
                    task=task,
                    task_emb=task_emb,
                    task_id=task_id,
                    sim_states=sim_states,
                    task_str=task_str,
                )
                successes.append(success_rate)

                if prev_success_rate < success_rate:
                    if self.cfg.lifelong.algo == "TaskSpecificParameter" or self.cfg.lifelong.algo == "ERID":
                        for idx, single_parameter in enumerate(self.policy.named_parameters()):
                            if "CP" in single_parameter[0] and single_parameter[0] not in self.shared_cp_weights:
                                self.current_task_specific_cp_weights[single_parameter[0]] = single_parameter[
                                    1].data.clone()
                        self.task_specific_cp_weights[
                                f"Task{self.current_task}"] = self.current_task_specific_cp_weights.copy()
                        print("[info] Found task_specific_cp_weights, save it")
                    print("[info] Found best performance model, save it")
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(bc_losses) - 1

                t1 = time.time()

                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                print(
                    f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.4f} ± {ci:4.4f} | best succ: {prev_success_rate:4.4f}"
                    + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.4f} | time: {(t1-t0)/60:4.2f}",
                    flush=True,
                )

                # early stop
                if prev_success_rate >= 0.95 and success_rate <= prev_success_rate:
                    success_rate_drop_time += 1
                    if success_rate_drop_time > 2:
                        break

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        # load the best performance agent on the current task
        self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])

        # end learning the current task, some algorithms need post-processing
        self.end_task(dataset, task_id, benchmark)

        # return the metrics regarding forward transfer
        bc_losses = np.array(bc_losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "loss": bc_losses,
            },
            auc_checkpoint_name,
        )

        # pretend that the agent stops learning once it reaches the peak performance
        bc_losses[idx_at_best_succ:] = bc_losses[idx_at_best_succ]
        successes[idx_at_best_succ:] = successes[idx_at_best_succ]
        return successes.sum() / cumulated_counter, bc_losses.sum() / cumulated_counter

    def reset(self):
        self.policy.reset()

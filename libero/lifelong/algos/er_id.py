import collections

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.utils import *


def cycle(dl):
    while True:
        for data in dl:
            yield data


def merge_datas(x, y):
    if isinstance(x, (dict, collections.OrderedDict)):
        if isinstance(x, dict):
            new_x = dict()
        else:
            new_x = collections.OrderedDict()

        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k])
        return new_x
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor):
        return torch.cat([x, y], 0)


class ERID(Sequential):
    """
    The experience replay policy.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        # we truncate each sequence dataset to a buffer, when replay is used,
        # concate all buffers to form a single replay buffer for replay.
        self.datasets = []
        self.descriptions = []
        self.buffer = None

        #  Task Specific & Global Parameter 模式分解变量
        self.task_specific_cp_weights = {}
        self.current_task_specific_cp_weights = {}
        self.shared_cp_weights = ['CP_U.weight', 'CP_V.weight']  # 'CP_C', 'CP_U.weight', 'CP_V.weight'

    def start_task(self, task):
        """
            对于每个新任务，都创建特定的分解参数、U V是共享的，P和λ是各自的
        """
        for idx, single_parameter in enumerate(self.policy.named_parameters()):
            if "CP" in single_parameter[0] and single_parameter[0] not in self.shared_cp_weights:
                single_parameter[1].data = nn.init.xavier_uniform_(
                    torch.zeros_like(single_parameter[1], requires_grad=True)).to(self.cfg.device)
                self.current_task_specific_cp_weights[single_parameter[0]] = single_parameter[1].data.clone()
        self.task_specific_cp_weights[f"Task{task}"] = self.current_task_specific_cp_weights.copy()

        super().start_task(task)
        if self.current_task > 0:
            # WARNING: currently we have a fixed size memory for each task.
            buffers = [
                TruncatedSequenceDataset(dataset, self.cfg.lifelong.n_memories)
                for dataset in self.datasets
            ]

            buf = ConcatDataset(buffers)
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=False,
                )
            )

    def end_task(self, dataset, task_id, benchmark):
        self.datasets.append(dataset)
        task_specific_cp_weights_pth = os.path.join(self.experiment_dir, "task_specific_cp_weights.npy")
        for idx, single_parameter in enumerate(self.policy.named_parameters()):
            if "CP" in single_parameter[0] and single_parameter[0] not in self.shared_cp_weights:
                self.current_task_specific_cp_weights[single_parameter[0]] = single_parameter[
                    1].data.clone()
        self.task_specific_cp_weights[
            f"Task{self.current_task}"] = self.current_task_specific_cp_weights.copy()
        print("[info] Found task_specific_cp_weights, save it")
        np.save(task_specific_cp_weights_pth, self.task_specific_cp_weights)

    def observe(self, data):
        if self.buffer is not None:
            buf_data = next(self.buffer)
            data = merge_datas(data, buf_data)

        data = self.map_tensor_to_device(data)
        prompt_loss = torch.zeros((1,), requires_grad=True)
        cp_c_loss = torch.zeros((1,), requires_grad=True)

        self.optimizer.zero_grad()
        if "Prompt" in self.cfg.policy["policy_type"] and "CP" in self.cfg.policy["policy_type"]:
            loss, prompt_loss, cp_c_loss = self.policy.cp_prompt_compute_loss(data)
        elif "CP" in self.cfg.policy["policy_type"]:
            loss, cp_c_loss = self.policy.cp_compute_loss(data)
        elif "Prompt" in self.cfg.policy["policy_type"]:
            loss, prompt_loss = self.policy.prompt_compute_loss(data)
        else:
            loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item(), prompt_loss.item(), cp_c_loss.item()

    def get_eval_algo(self, task_id):
        # TODO: find a better way to do this
        # save and load a new model and set all params where mask > current_task + 1 to 0
        torch_save_model(
            self.policy,
            os.path.join(self.experiment_dir, "tmp_model.pth"),
            cfg=self.cfg,
        )
        eval_algo = safe_device(
            eval(self.cfg.lifelong.algo)(
                self.n_tasks, self.cfg
            ),
            self.cfg.device,
        )
        model_state_dict, _, _ = torch_load_model(
            os.path.join(self.experiment_dir, "tmp_model.pth")
        )
        eval_algo.policy.load_state_dict(model_state_dict)
        eval_algo.optimizer = self.optimizer
        eval_algo.scheduler = self.scheduler
        eval_algo.experiment_dir = self.experiment_dir
        eval_algo.task_specific_cp_weights = self.task_specific_cp_weights
        eval_algo.current_task_specific_cp_weights = self.current_task_specific_cp_weights
        eval_algo.shared_cp_weights = self.shared_cp_weights

        eval_algo.eval()

        if not self.policy.training:
            print(f"[info] loading task {task_id:3d} best specific parameters...")
            for idx, single_parameter in enumerate(eval_algo.policy.named_parameters()):
                if "CP" in single_parameter[0] and single_parameter[0] not in self.shared_cp_weights:
                    single_parameter[1].data = (self.task_specific_cp_weights[f"Task{task_id}"][single_parameter[0]]
                                                .to(self.cfg.device))
        return eval_algo

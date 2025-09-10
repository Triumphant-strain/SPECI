import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libero.libero.benchmark import *
from libero.lifelong.algos.base import Sequential
from libero.lifelong.metric import *
from libero.lifelong.utils import *


class TaskSpecificParameter(Sequential):
    """
    My Task Specific Parameter policy.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)

        previous_masks = {}
        for module_idx, module in enumerate(self.policy.named_modules()):
            if (isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear) and
                    "CP" not in module[0]):
                mask = torch.ByteTensor(module[1].weight.data.size()).fill_(0)
                if "cuda" in module[1].weight.data.type():
                    mask = mask.to(self.cfg.device)
                previous_masks[module_idx] = mask
        self.previous_masks = previous_masks
        #  Task Specific & Global Parameter 模式分解变量
        self.task_specific_cp_weights = {}
        self.current_task_specific_cp_weights = {}
        self.shared_cp_weights = ['CP_U.weight', 'CP_V.weight']  # 'CP_C', 'CP_U.weight', 'CP_V.weight'

    def pruning_mask(self, weights, previous_mask, layer_idx):
        """
        Ranks weights by magnitude. Sets all below kth to 0.
        Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.to(self.cfg.device)
        tensor = weights[  # 找到属于当前任务的权重
            previous_mask.eq(self.current_task + 1)
        ]  # current_task starts from 0, so we add 1
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.cfg.lifelong.prune_perc * tensor.numel())  # 要把权重矩阵中所有元素的百分之多少prune掉，即留给新任务
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]  # kthvalue()函数返回第k小的值及其索引

        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(
            self.current_task + 1
        )  # le()函数执行逐元素的小于等于比较，返回一个布尔张量，指示哪些权重的绝对值小于等于剪枝阈值。

        # mask is 1 - remove_mask
        previous_mask[remove_mask.eq(1)] = 0
        mask = previous_mask
        print(
            "Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)"
            % (
                layer_idx,
                mask.eq(0).sum(),
                tensor.numel(),
                100 * mask.eq(0).sum() / tensor.numel(),
                weights.numel(),
            )
        )
        return mask

    def prune(self):
        """
        Gets pruning mask for each layer, based on previous_masks.
        Sets the self.current_masks to the computed pruning masks.
        """
        self.current_masks = {}
        print(
            "[info] pruning each layer by removing %.2f%% of values"
            % (100 * self.cfg.lifelong.prune_perc)
        )

        for module_idx, module in enumerate(self.policy.named_modules()):
            if (isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear) and
                    "CP" not in module[0]):
                mask = self.pruning_mask(
                    module[1].weight.data, self.previous_masks[module_idx], module_idx
                )
                self.current_masks[module_idx] = mask.to(self.cfg.device)

                # Set pruned weights to 0.
                # TODO(xjk): 是否要使用之前训练的参数，而不是0
                weight = module[1].weight.data
                weight[self.current_masks[module_idx].eq(0)] = 0.0
        self.previous_masks = self.current_masks

    def make_grads_zero(self):
        """
        Sets grads of fixed weights and Norm layers to 0 and save current task specific weights
        """
        # for idx, single_parameter in enumerate(self.policy.named_parameters()):
        #     if "CP" in single_parameter[0] and single_parameter[0] not in self.shared_cp_weights:
        #         self.current_task_specific_cp_weights[single_parameter[0]] = single_parameter[1].data.clone()
        # self.task_specific_cp_weights[f"Task{self.current_task}"] = self.current_task_specific_cp_weights

        assert self.current_masks

        for module_idx, module in enumerate(self.policy.named_modules()):
            if (isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear) and
                    "CP" not in module[0]):
                layer_mask = self.current_masks[module_idx]

                # Set grads of all weights not belonging to current dataset to 0.
                if module[1].weight.grad is not None:
                    # ne()函数执行逐元素的不等于比较，返回一个布尔张量，指示哪些位置的任务索引与当前任务索引不相等。
                    module[1].weight.grad.data[layer_mask.ne(self.current_task + 1)] = 0
                    # Biases are fixed.
                    if module[1].bias is not None:
                        module[1].bias.grad.data.fill_(0)
            elif "BatchNorm" in str(type(module[1])) or "LayerNorm" in str(type(module[1])):
                # Set grads of batchnorm params to 0.
                module[1].weight.grad.data.fill_(0)
                module[1].bias.grad.data.fill_(0)

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

        super().start_task(task)  # 初始化优化器和规划器
        assert self.previous_masks
        for module_idx, module in enumerate(self.policy.named_modules()):
            if (isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear) and
                    "CP" not in module[0]):
                mask = self.previous_masks[module_idx]
                # since current_task starts from 0, we add 1
                mask[mask.eq(0)] = self.current_task + 1  # 修改previous_masks[module_idx]里等于0的值为当前任务id，即可学习的参数
            # we never train norm layers
            elif "BatchNorm" in str(type(module[1])) or "LayerNorm" in str(type(module[1])):
                module[1].eval()

        self.current_masks = self.previous_masks

    def observe(self, data):
        # make norm layer to eval
        for module_idx, module in enumerate(self.policy.modules()):
            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()
        data = self.map_tensor_to_device(data)
        prompt_loss = torch.zeros((1,), requires_grad=True)
        cp_c_loss = torch.zeros((1,), requires_grad=False)

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
        (loss * self.loss_scale).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )

        self.make_grads_zero()
        self.optimizer.step()
        return loss.item(), prompt_loss.item(), cp_c_loss.item()
        # return loss.item(), prompt_loss.item()

    def end_task(self, dataset, task_id, benchmark):
        """prune + post_finetune
        for fair comparisons with other lifelong learning algorithms,
        we do not use the success rates in the post_finetune epochs to AUC"""
        self.prune()

        # Do final fine-tuning to improve results on pruned network.
        if self.cfg.lifelong.post_prune_epochs:
            print("[info] start finetuning after pruning ...")
            # Note: here we do not apply start_task() to keep the 0 value in the
            # mask stay 0 and only update the param where mask=current_task+1
            # re-initialize the optimizer and scheduler
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

            model_checkpoint_name = os.path.join(
                self.experiment_dir, f"task{task_id}_model.pth"
            )

            task_specific_cp_weights_pth = os.path.join(self.experiment_dir, "task_specific_cp_weights.npy")

            train_dataloader = DataLoader(
                dataset,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers,
                shuffle=True,
            )

            prev_success_rate = -1.0
            success_rate_drop_time = 0
            best_state_dict = self.policy.state_dict()  # currently save the best model
            torch_save_model(
                self.policy, model_checkpoint_name, cfg=self.cfg,
                previous_masks=self.previous_masks,
            )

            # this is just a fake summary object that works for placeholders
            sim_states = [[] for _ in range(self.cfg.eval.n_eval)]
            for epoch in range(0, self.cfg.lifelong.post_prune_epochs + 1):
                t0 = time.time()
                self.policy.train()
                training_loss = 0.0
                # training_att_sparsity_loss = 0.0
                # training_codebook_normalize_loss = 0.0
                training_prompt_loss = 0.0
                training_cp_c_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    # loss, att_sparsity_loss, codebook_normalize_loss, prompt_loss = self.observe(data)
                    # loss, prompt_loss = self.observe(data)
                    loss, prompt_loss, cp_c_loss = self.observe(data)
                    training_loss += loss
                    # training_att_sparsity_loss += att_sparsity_loss
                    # training_codebook_normalize_loss += codebook_normalize_loss
                    training_prompt_loss += prompt_loss
                    training_cp_c_loss += cp_c_loss
                training_loss /= len(train_dataloader)
                # training_att_sparsity_loss /= len(train_dataloader)
                # training_codebook_normalize_loss /= len(train_dataloader)
                training_prompt_loss /= len(train_dataloader)
                training_cp_c_loss /= len(train_dataloader)
                t1 = time.time()
                if "Prompt" and "CP" in self.cfg.policy["policy_type"]:
                    # print(
                    #     f"[info] Epoch: {epoch:3d} | total train loss: {training_loss:5.4f} | "
                    #     f"prompt_loss: {training_prompt_loss:5.4f} | cp_c_loss: {training_cp_c_loss:5.4f} | "
                    #     f"time: {(t1 - t0) / 60:4.2f}"
                    # )
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
                        f"time: {(t1 - t0) / 60:4.2f}")

                time.sleep(0.1)

                if epoch % self.cfg.lifelong.post_eval_every == 0:  # evaluate BC loss
                    # self.policy.eval()

                    t0 = time.time()
                    task = benchmark.get_task(task_id)
                    task_emb = benchmark.get_task_emb(task_id)
                    task_str = f"k{task_id}_e{epoch // self.cfg.lifelong.post_eval_every}"

                    success_rate = evaluate_one_task_success(
                        self.cfg,
                        self,
                        task,
                        task_emb,
                        task_id,
                        sim_states=sim_states,
                        task_str=task_str,
                    )

                    if prev_success_rate < success_rate:
                        # we do not record the success rate
                        for idx, single_parameter in enumerate(self.policy.named_parameters()):
                            if "CP" in single_parameter[0] and single_parameter[0] not in self.shared_cp_weights:
                                self.current_task_specific_cp_weights[single_parameter[0]] = single_parameter[
                                    1].data.clone()
                        self.task_specific_cp_weights[
                                f"Task{self.current_task}"] = self.current_task_specific_cp_weights.copy()
                        print("[info] Found task_specific_cp_weights, save it")
                        print("[info] Found best performance model, save it")
                        torch_save_model(
                            self.policy, model_checkpoint_name, cfg=self.cfg,
                            previous_masks=self.previous_masks,
                        )
                        prev_success_rate = success_rate

                    t1 = time.time()

                    ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                    print(
                        f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f}"
                        + f"| best succ: {prev_success_rate} "
                        + f"| time: {(t1 - t0) / 60:4.2f}"
                    )

                    # early stop
                    if prev_success_rate >= 0.95 and success_rate <= prev_success_rate:
                        success_rate_drop_time += 1
                        if success_rate_drop_time > 2:
                            break

                if self.scheduler is not None:
                    self.scheduler.step()

            self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])
            np.save(task_specific_cp_weights_pth, self.task_specific_cp_weights)
            self.policy.eval()

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

        eval_algo.previous_masks = self.previous_masks
        eval_algo.pruning_mask = self.pruning_mask
        eval_algo.current_masks = self.current_masks
        eval_algo.current_task = self.current_task
        eval_algo.optimizer = self.optimizer
        eval_algo.scheduler = self.scheduler
        eval_algo.experiment_dir = self.experiment_dir
        eval_algo.task_specific_cp_weights = self.task_specific_cp_weights
        eval_algo.current_task_specific_cp_weights = self.current_task_specific_cp_weights
        eval_algo.shared_cp_weights = self.shared_cp_weights


        eval_algo.eval()

        for module_idx, module in enumerate(eval_algo.policy.named_modules()):
            if (isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear) and
                    "CP" not in module[0]):
                weight = module[1].weight.data
                mask = eval_algo.previous_masks[module_idx].to(self.cfg.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(task_id + 1)] = 0.0
            # we never train norm layers
            elif "BatchNorm" in str(type(module[1])) or "LayerNorm" in str(type(module[1])):
                module[1].eval()

        if not self.policy.training:
            print(f"[info] loading task {task_id:3d} best specific parameters...")
            for idx, single_parameter in enumerate(eval_algo.policy.named_parameters()):
                if "CP" in single_parameter[0] and single_parameter[0] not in self.shared_cp_weights:
                    single_parameter[1].data = (self.task_specific_cp_weights[f"Task{task_id}"][single_parameter[0]]
                                                .to(self.cfg.device))
        return eval_algo

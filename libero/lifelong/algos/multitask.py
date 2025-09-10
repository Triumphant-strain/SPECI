import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *


class Multitask(Sequential):
    """
    The multitask learning baseline/upperbound.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)

    def learn_all_tasks(self, datasets, benchmark, result_summary):
        self.start_task(-1)
        concat_dataset = ConcatDataset(datasets)

        # learn on all tasks, only used in multitask learning
        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_model.pth"
        )
        all_tasks = list(range(benchmark.n_tasks))

        train_dataloader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(concat_dataset),
            # persistent_workers=True,
        )

        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict()  # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        bc_losses = []

        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()
            training_loss = 0.0
            # training_att_sparsity_loss = 0.0
            # training_codebook_normalize_loss = 0.0
            training_prompt_loss = 0.0
            training_cp_c_loss = 0.0
            if epoch > 0 or self.cfg.pretrain:  # update
                self.policy.train()
                for (idx, data) in enumerate(train_dataloader):
                    # loss, att_sparsity_loss, codebook_normalize_loss = self.observe(data)
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
            else:  # just evaluate the zero-shot performance on 0-th epoch
                for (idx, data) in enumerate(train_dataloader):
                    # loss, att_sparsity_loss, codebook_normalize_loss = self.eval_observe(data)
                    # loss, prompt_loss = self.eval_observe(data)
                    loss, prompt_loss, cp_c_loss = self.eval_observe(data)
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

            # print(
            #     f"[info] Epoch: {epoch:3d} | total train loss: {training_loss:5.2f} | "
            #     f"att_sparsity_loss: {training_att_sparsity_loss:5.2f} | "
            #     f"codebook_normalize_loss: {training_codebook_normalize_loss:5.2f} | time: {(t1 - t0) / 60:4.2f}"
            # )
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

            if epoch % self.cfg.eval.eval_every == 0:  # evaluate BC loss
                # bc_losses.append(training_loss - training_att_sparsity_loss - training_codebook_normalize_loss)
                # bc_losses.append(training_loss - training_prompt_loss)
                bc_losses.append(training_loss - training_prompt_loss - training_cp_c_loss)
                t0 = time.time()
                self.policy.eval()

                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"multitask_model_ep{epoch}.pth"
                )
                torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)

                # for multitask learning, we provide an option whether to evaluate
                # the agent once every eval_every epochs on all tasks, note that
                # this can be quite computationally expensive. Nevertheless, we
                # save the checkpoints, so users can always evaluate afterwards.
                if self.cfg.lifelong.eval_in_train:
                    success_rates = evaluate_multitask_training_success(
                        self.cfg, self, benchmark, all_tasks
                    )
                    success_rate = np.mean(success_rates)
                else:
                    success_rate = 0.0
                successes.append(success_rate)

                if prev_success_rate < success_rate and (not self.cfg.pretrain):
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(bc_losses) - 1

                t1 = time.time()

                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]

                if self.cfg.lifelong.eval_in_train:
                    print(
                        f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.4f} ± {ci:4.4f} | best succ: {prev_success_rate:4.4f} "
                        + f"| succ. AoC {tmp_successes.sum() / cumulated_counter:4.4f} | time: {(t1 - t0) / 60:4.2f}",
                        flush=True,
                    )

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        # load the best policy if there is any
        if self.cfg.lifelong.eval_in_train:
            self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])
        self.end_task(concat_dataset, -1, benchmark)

        # return the metrics regarding forward transfer
        bc_losses = np.array(bc_losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "loss": bc_losses,
            },
            auc_checkpoint_name,
        )

        if self.cfg.lifelong.eval_in_train:
            loss_at_best_succ = bc_losses[idx_at_best_succ]
            success_at_best_succ = successes[idx_at_best_succ]
            bc_losses[idx_at_best_succ:] = loss_at_best_succ
            successes[idx_at_best_succ:] = success_at_best_succ
        return successes.sum() / cumulated_counter, bc_losses.sum() / cumulated_counter

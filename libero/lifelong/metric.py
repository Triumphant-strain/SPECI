import copy
import gc
import imageio
import numpy as np
import os
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.utils import *


def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)
    data = {
        "obs": {},
        "task_emb": task_emb.repeat(env_num, 1),
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["obs"][obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][cfg.data.obs_key_mapping[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    return data


def evaluate_one_task_success(
        cfg, algo, task, task_emb, task_id, sim_states=None, task_str=""
):
    """
    Evaluate a single task's success rate
    sim_states: if not None, will keep track of all simulated states during
                evaluation, mainly for visualization and debugging purpose
    task_str:   the key to access sim_states dictionary
    """
    with Timer() as t:
        # need preprocess weights for PackNet and TaskSpecificParameter
        if cfg.lifelong.algo == "PackNet" or cfg.lifelong.algo == "TaskSpecificParameter":
            algo = algo.get_eval_algo(task_id)

        algo.eval()

        # initiate evaluation envs
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # Try to handle the frame buffer issue
        env_creation = False

        count = 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    env = DummyVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                else:
                    env = SubprocVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")

        #  Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        num_success = 0
        for i in range(eval_loop_num):
            env.reset()
            indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            init_states_ = init_states[indices]

            dones = [False] * env_num
            steps = 0
            algo.reset()
            obs = env.set_init_state(init_states_)

            # dummy actions [env_num, 7] all zeros for initial physics simulation
            dummy = np.zeros((env_num, 7))
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            if task_str != "":
                sim_state = env.get_sim_state()  # (1, 92)
                for k in range(env_num):
                    if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                        sim_states[i * env_num + k].append(sim_state[k])

            obs_tensors = [[]] * env_num  # # obs_tensor: (env_num, T, H, W, C)
            while steps < cfg.eval.max_steps:
                steps += 1
                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)  # data里包括obs(eye in hand rgb; gripper states; joint
                # states; agent view rgb)和task_embedding
                # print(data['obs']['agentview_rgb'].shape)  # Tensor (1, 3, 128, 128)
                actions = algo.policy.get_action(data)  # 已修改：原来是algo.policy.get_action(data)

                obs, reward, done, info = env.step(actions)

                # record the sim states for replay purpose
                if task_str != "":
                    sim_state = env.get_sim_state()
                    for k in range(env_num):
                        if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                            sim_states[i * env_num + k].append(sim_state[k])

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                    obs_tensors[k].append(obs[k]["agentview_image"])

                if all(dones):
                    break

            if not os.path.exists(os.path.join(cfg.experiment_dir, task_str)):
                os.makedirs(os.path.join(cfg.experiment_dir, task_str))

            # a new form of success record
            for k in range(env_num):
                images = [img[::-1] for img in obs_tensors[k]]
                video_writer = imageio.get_writer(uri=os.path.join(cfg.experiment_dir,
                                                                   task_str, f"env_{k}_simulation_{i}.mp4"), fps=30)
                for image in images:
                    video_writer.append_data(image)
                video_writer.close()
                if i * env_num + k < cfg.eval.n_eval:
                    num_success += int(dones[k])

        success_rate = num_success / cfg.eval.n_eval
        env.close()
        gc.collect()
    print(f"[info] evaluate task {task_id} takes {t.get_elapsed_time():.1f} seconds")
    return success_rate


def evaluate_success(cfg, algo, benchmark, task_ids, result_summary=None):
    """
    Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        task_str = f"k{task_ids[-1]}_p{i}"  # k代表刚刚学过的任务，p表示当前要测试的任务
        curr_summary = result_summary[task_str] if result_summary is not None else None
        success_rate = evaluate_one_task_success(
            cfg, algo, task_i, task_emb, i, sim_states=curr_summary, task_str=task_str
        )
        successes.append(success_rate)
    return np.array(successes)


def evaluate_multitask_training_success(cfg, algo, benchmark, task_ids):
    """
    Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        success_rate = evaluate_one_task_success(cfg, algo, task_i, task_emb, i)
        successes.append(success_rate)
    return np.array(successes)


def get_auc(experiment_dir, cfg, N_TASKS=10, N_SEEDS=1, seeds=[10000]):
    N_EP = cfg.train.n_epochs // cfg.eval.eval_every + 1  # 11
    fwds = np.zeros((N_TASKS, N_EP, N_SEEDS))

    for task in range(N_TASKS):
        counter = 0
        for k, seed in enumerate(seeds):
            name = f"{experiment_dir}/task{task}_auc.log"
            try:
                succ = torch.load(name)["success"]  # (n_epochs)
                idx = succ.argmax()
                succ[idx:] = succ[idx]
                if succ.shape[0] < N_EP:  # 因为Early Stop导致测试的轮次可能不够11次，所以按照最高准确率补一下
                    for i in range(N_EP - succ.shape[0]):
                        succ = np.append(succ, succ[idx])
                fwds[task, :, k] = succ
            except:
                print("Some errors when loading results")
                print('Name: ', name)
                continue
    return fwds


def compute_metric(res):
    mat, fwts = res  # fwds: (num_tasks, num_save_intervals, num_seeds)
    num_tasks, num_seeds = mat.shape[1:]
    ret = {}

    # compute fwt
    fwt = fwts.mean(axis=(0, 1))
    ret["fwt"] = fwt
    # compute bwt
    bwts = []
    aucs = []
    for seed in range(num_seeds):
        bwt = 0.0
        auc = 0.0
        for k in range(num_tasks):
            bwt_k = 0.0
            auc_k = 0.0
            for tau in range(k + 1, num_tasks):
                bwt_k += mat[k, k, seed] - mat[tau, k, seed]
                auc_k += mat[tau, k, seed]
            if k + 1 < num_tasks:
                bwt_k /= (num_tasks - k - 1)
            auc_k = (auc_k + fwts[k, :, seed].mean()) / (num_tasks - k)

            bwt += bwt_k
            auc += auc_k
        bwts.append(bwt / num_tasks)
        aucs.append(auc / num_tasks)
    bwts = np.array(bwts)
    aucs = np.array(aucs)
    ret["bwt"] = bwts
    ret["auc"] = aucs
    return ret


@torch.no_grad()
def evaluate_loss(cfg, algo, benchmark, datasets):
    """
    Evaluate the loss on all datasets.
    """
    algo.eval()
    total_losses = []
    att_sparsity_losses = []
    codebook_normalize_losses = []
    prompt_losses = []
    cp_c_losses = []
    for i, dataset in enumerate(datasets):
        # need preprocess weights for PackNet and TaskSpecificParameter
        if cfg.lifelong.algo == "PackNet" or cfg.lifelong.algo == "TaskSpecificParameter" or cfg.lifelong.algo == "ERID":
            algo = algo.get_eval_algo(task_id=i)

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            shuffle=False,
            persistent_workers=False,
        )
        test_loss = 0
        test_prompt_loss = 0.0
        test_cp_c_loss = 0.0

        for data in dataloader:
            data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=cfg.device)
            )
            prompt_loss = torch.tensor(0.0)
            cp_c_loss = torch.tensor(0.0)
            if "WOID" in cfg.policy["policy_type"]:
                loss, prompt_loss, cp_c_loss = algo.policy.cp_prompt_woid_compute_loss(data)
            elif "Prompt" in cfg.policy["policy_type"] and "CP" in cfg.policy["policy_type"]:
                loss, prompt_loss = algo.policy.cp_prompt_compute_loss(data)
            elif "CP" in cfg.policy["policy_type"]:
                loss = algo.policy.cp_compute_loss(data)
            elif "Prompt" in cfg.policy["policy_type"]:
                loss, prompt_loss = algo.policy.prompt_compute_loss(data)
            else:
                loss = algo.policy.compute_loss(data)

            test_loss += loss.item()
            test_prompt_loss += prompt_loss.item()
            test_cp_c_loss += cp_c_loss.item()

        test_loss /= len(dataloader)
        test_prompt_loss /= len(dataloader)
        test_cp_c_loss /= len(dataloader)

        total_losses.append(test_loss)
        prompt_losses.append(test_prompt_loss)
        cp_c_losses.append(test_cp_c_loss)

    return np.array(total_losses), np.array(prompt_losses), np.array(cp_c_losses)
    # return np.array(total_losses), np.array(prompt_losses)

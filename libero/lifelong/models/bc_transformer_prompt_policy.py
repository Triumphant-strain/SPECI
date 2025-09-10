import copy

import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *


###############################################################################
#
# A model handling extra input modalities besides images at time t.
#
###############################################################################


class ExtraModalityTokens(nn.Module):
    def __init__(
            self,
            use_joint=False,
            use_gripper=False,
            use_ee=False,
            extra_num_layers=0,
            extra_hidden_size=64,
            extra_embedding_size=32,
    ):
        """
        This is a class that maps all extra modality inputs into tokens of the same size
        """
        super().__init__()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3

        self.num_extra = int(use_joint) + int(use_gripper) + int(use_ee)

        extra_low_level_feature_dim = (
                int(use_joint) * joint_states_dim
                + int(use_gripper) * gripper_states_dim
                + int(use_ee) * ee_dim
        )

        assert extra_low_level_feature_dim > 0, "[error] no extra information"

        self.extra_encoders = {}

        def generate_proprio_mlp_fn(modality_name, extra_low_level_feature_dim):
            assert extra_low_level_feature_dim > 0  # we indeed have extra information
            if extra_num_layers > 0:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_hidden_size)]
                for i in range(1, extra_num_layers):
                    layers += [
                        nn.Linear(extra_hidden_size, extra_hidden_size),
                        nn.ReLU(inplace=True),
                    ]
                layers += [nn.Linear(extra_hidden_size, extra_embedding_size)]
            else:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_embedding_size)]

            self.proprio_mlp = nn.Sequential(*layers)
            self.extra_encoders[modality_name] = {"encoder": self.proprio_mlp}

        for (proprio_dim, use_modality, modality_name) in [
            (joint_states_dim, self.use_joint, "joint_states"),
            (gripper_states_dim, self.use_gripper, "gripper_states"),
            (ee_dim, self.use_ee, "ee_states"),
        ]:

            if use_modality:
                generate_proprio_mlp_fn(modality_name, proprio_dim)

        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.extra_encoders.values()]
        )

    def forward(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3)
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []

        for (use_modality, modality_name) in [
            (self.use_joint, "joint_states"),
            (self.use_gripper, "gripper_states"),
            (self.use_ee, "ee_states"),
        ]:

            if use_modality:
                tensor_list.append(
                    self.extra_encoders[modality_name]["encoder"](
                        obs_dict[modality_name]
                    )
                )

        x = torch.stack(tensor_list, dim=-2)
        return x


class PerturbationAttention:
    """
    See https://arxiv.org/pdf/1711.00138.pdf for perturbation-based visualization
    for understanding a control agent.
    """

    def __init__(self, model, image_size=[128, 128], patch_size=[16, 16], device="cpu"):
        self.model = model
        self.patch_size = patch_size
        H, W = image_size
        num_patches = (H * W) // np.prod(patch_size)
        # pre-compute mask
        h, w = patch_size
        nh, nw = H // h, W // w
        mask = (
            torch.eye(num_patches)
            .view(num_patches, num_patches, 1, 1)
            .repeat(1, 1, patch_size[0], patch_size[1])
        )  # (np, np, h, w)
        mask = rearrange(
            mask.view(num_patches, nh, nw, h, w), "a b c d e -> a (b d) (c e)"
        )  # (np, H, W)
        self.mask = mask.to(device).view(1, num_patches, 1, H, W)
        self.num_patches = num_patches
        self.H, self.W = H, W
        self.nh, self.nw = nh, nw

    def __call__(self, data):
        rgb = data["obs"]["agentview_rgb"]  # (B, C, H, W)
        B, C, H, W = rgb.shape

        rgb_ = rgb.unsqueeze(1).repeat(1, self.num_patches, 1, 1, 1)  # (B, np, C, H, W)
        rgb_mean = rgb.mean([2, 3], keepdims=True).unsqueeze(1)  # (B, 1, C, 1, 1)
        rgb_new = (rgb_mean * self.mask) + (1 - self.mask) * rgb_  # (B, np, C, H, W)
        rgb_stack = torch.cat([rgb.unsqueeze(1), rgb_new], 1)  # (B, 1+np, C, H, W)

        rgb_stack = rearrange(rgb_stack, "b n c h w -> (b n) c h w")
        res = self.model(rgb_stack).view(B, self.num_patches + 1, -1)  # (B, 1+np, E)
        base = res[:, 0].view(B, 1, -1)
        others = res[:, 1:].view(B, self.num_patches, -1)

        attn = F.softmax(1e5 * (others - base).pow(2).sum(-1), -1)  # (B, num_patches)
        attn_ = attn.view(B, 1, self.nh, self.nw)
        attn_ = (
            F.interpolate(attn_, size=(self.H, self.W), mode="bilinear")
            .detach()
            .cpu()
            .numpy()
        )
        return attn_


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


def weight_spa_loss(alpha):
    loss = 0
    for key in range(alpha.shape[0]):
        loss = loss - alpha[key].max(dim=1)[0].mean()
    loss = loss / len(alpha)
    return loss


class CodaPrompt(nn.Module):
    def __init__(self, policy_cfg):
        super().__init__()
        self.task_count = 0
        self.emb_d = policy_cfg.embed_size
        self.key_d = policy_cfg.key_size
        self.n_tasks = policy_cfg.n_tasks
        self._init_smart(policy_cfg.embed_size, policy_cfg.prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full parameters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, policy_cfg.embed_size * policy_cfg.transformer_num_heads)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])  # M
        self.e_p_length = int(prompt_param[1])  # l_p
        self.e_layers = [0, 1, 2, 3]

        # strength of ortho penalty
        self.ortho_mu = prompt_param[2]

    def process_task_count(self, task_id):
        self.task_count = task_id

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more
        # fair in the spirit of continual learning and has little affect on performance
        #
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self, f'e_k_{e}')
            A = getattr(self, f'e_a_{e}')
            P = getattr(self, f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):  # 计算向量v在向量u上的投影
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)  # e_pool_size
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / self.n_tasks)
        s = int(self.task_count * pt)  # 该任务prompt的开头
        f = int((self.task_count + 1) * pt)  # 该任务prompt的末尾
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()  # 将vv的前s列（表示之前任务的子技能库）复制到uu的相应位置。
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)  # 标准正态分布初始化
                uk = 0
                for j in range(0, k):  # 遍历所有先前任务的子技能库的技能向量
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)  # 把uj按vk在uj上的投影的值给缩放了，其实计算的就是施密特正交化需要减去的值
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            # B, C = x_querry.shape

            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')
            pt = int(self.e_pool_size / self.n_tasks)
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:  # 不更新属于之前任务的prompt
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]  # K torch.Size([10, 64])
                A = A[0:f]  # A torch.Size([10, 64])
                p = p[0:f]  # p torch.Size([10, 12, 64])

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            batch_A = torch.stack([A.clone() for _ in range(x_querry.shape[0])], dim=0)
            a_querry = torch.einsum('Bbd,Bkd->Bbkd', x_querry, batch_A)  # b:50, k:10
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            batch_n_K = torch.stack([n_K.clone() for _ in range(x_block.shape[0])], dim=0)
            q = nn.functional.normalize(a_querry, dim=3)
            aq_k = torch.einsum('Bbkd,Bkd->Bbk', q, batch_n_K)  # 余弦相似度

            # 在最后一个维度上获取前10大的值及其索引
            top10_values, top10_indices = torch.topk(aq_k, 10, dim=-1)
            # 对前10大的值进行softmax
            top10_softmax = F.softmax(top10_values, dim=-1)
            # 创建一个全零张量
            aq_k = torch.zeros_like(aq_k)
            # 将softmax后的值放入原始张量相应的位置
            aq_k.scatter_(-1, top10_indices, top10_softmax)

            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            batch_p = torch.stack([p.clone() for _ in range(x_block.shape[0])], dim=0)
            P_ = torch.einsum('Bbk,Bkld->Bbld', aq_k, batch_p)  # aq_k即alpha，加权 (B, 50, e_p_length, embed_size)

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :, :i, :]
            Ev = P_[:, :, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()


###############################################################################
#
# A Hierarchical Transformer Policy
#
###############################################################################

class BCTransformerPromptPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        self.R = policy_cfg.R
        embed_size = policy_cfg.embed_size
        self.n_tasks = policy_cfg.n_tasks

        # 1. encode image
        transformer_input_sizes = []
        self.image_encoders = {}
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = embed_size
                kwargs.language_dim = (
                    policy_cfg.language_encoder.network_kwargs.input_size
                )
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                }

        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )

        #  2. encode language
        policy_cfg.language_encoder.network_kwargs.output_size = embed_size
        self.language_encoder = eval(policy_cfg.language_encoder.network)(
            **policy_cfg.language_encoder.network_kwargs
        )

        #  3. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalityTokens(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_hidden_size,
            extra_embedding_size=embed_size,
        )

        #  4. define temporal transformers
        policy_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
        self.temporal_position_encoding_fn = eval(
            policy_cfg.temporal_position_encoding.network
        )(**policy_cfg.temporal_position_encoding.network_kwargs)

        self.temporal_transformer_skill = TransformerDecoder_Skill_Prompt(
            input_size=embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
            R=self.R
        )

        # self.temporal_transformer_action = CP_TransformerDecoder_Action(
        #     input_size=embed_size,
        #     num_layers=policy_cfg.transformer_num_layers,
        #     num_heads=policy_cfg.transformer_num_heads,
        #     head_output_size=policy_cfg.transformer_head_output_size,
        #     mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
        #     dropout=policy_cfg.transformer_dropout,
        #     R=self.R,
        # )

        # 5. define prompt
        self.prompt = CodaPrompt(policy_cfg)
        self.prompt_param = policy_cfg.prompt_param

        # 6. define policy head
        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = embed_size
        policy_head_kwargs.output_size = shape_meta["ac_dim"]

        self.policy_head = eval(policy_cfg.policy_head.network)(
            **policy_cfg.policy_head.loss_kwargs,
            **policy_cfg.policy_head.network_kwargs
        )

        self.latent_queue = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

    def skill_inference_decode(self, x):
        pos_emb = self.temporal_position_encoding_fn(x)  # (T:10, E:384)
        x = x + pos_emb.unsqueeze(1)  # (B:32, T:10, num_modality:5, E:384)
        sh = x.shape
        self.temporal_transformer_skill.compute_mask(x.shape, [x.shape[0], x.shape[1] * x.shape[2] * 2, x.shape[3]])
        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        x, prompt_loss = self.temporal_transformer_skill(x, self.prompt)
        x = x.reshape(*sh)  # (B, T:10, num_modalities:5, E:384)
        return x[:, :, 0], prompt_loss  # (B, T, E)

    # def action_execute_decode(self, x):
    #     # print("x: ", x.shape)  (1, 10, E)
    #     pos_emb = self.temporal_position_encoding_fn(x)  # (10, E)
    #     x = x.unsqueeze(2) + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
    #     sh = x.shape  # (1, 10, 1, 64)
    #     self.temporal_transformer_action.compute_mask(x.shape)
    #
    #     x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
    #     x = self.temporal_transformer_action(x, CP_U=self.CP_U, CP_V=self.CP_V, CP_C=self.CP_C)
    #     x = x.reshape(*sh)
    #     return x[:, :, 0]  # (B, T, E)

    def spatial_encode(self, data):
        # 1. encode extra
        extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E)

        # 2. encode language, treat it as action token
        B, T = extra.shape[:2]
        text_encoded = self.language_encoder(data)  # (B, E)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (B, T, 1, E)
        encoded = [text_encoded, extra]

        # 3. encode image
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            img_encoded = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, 1, -1)
            encoded.append(img_encoded)
        encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
        return encoded

    def forward(self, data):
        x = self.spatial_encode(data)  # torch.Size([B:32, T:10, num_modalities:5, E])
        x, prompt_loss = self.skill_inference_decode(x)  # (B:32, T:10, E)
        # print("after skill_inference_decode shape: ", x.shape)
        # x = self.action_execute_decode(x)
        # print("after action_execute_decode shape: ", x.shape)   (B, T, E)
        dist = self.policy_head(x)  # .sample()能够得到动作 [B:32, T:10, ac_dim:7]
        return dist, prompt_loss

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            x = self.spatial_encode(data)
            self.latent_queue.append(x)
            if len(self.latent_queue) < self.max_seq_len:  # 确保输入是10个时间点的序列,当前做法是将初始状态复制10份,作为第一次的输入
                for i in range(self.max_seq_len - 1):
                    self.latent_queue.append(x)
            if len(self.latent_queue) > self.max_seq_len:  # 随着策略的执行和环境状态的变化,将之前的状态删掉,维持一组10个时间点
                self.latent_queue.pop(0)
            x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
            x, _ = self.skill_inference_decode(x)
            # x = self.action_execute_decode(x)
            dist = self.policy_head(x[:, -1])  # 选取了最后一个时间步
        action = dist.sample().detach().cpu()  # 动作采样 (T:1, ac_dim:7)
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.latent_queue = []

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


###############################################################################
#
# A Hierarchical Transformer Policy
#
###############################################################################

class BCHierarchicalTransformerCPPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        self.R = policy_cfg.R
        embed_size = policy_cfg.embed_size

        self.CP_U = nn.Linear(in_features=embed_size, out_features=self.R, bias=False)
        self.CP_V = nn.Linear(self.R, out_features=embed_size, bias=False)
        self.CP_C = nn.Parameter(torch.zeros([self.R, self.R, self.R], dtype=torch.float), requires_grad=True)
        nn.init.zeros_(self.CP_V.weight)  # only one the U and V need to be zero
        nn.init.xavier_uniform_(self.CP_C)

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

        self.temporal_transformer_skill = CP_TransformerDecoder_Skill(
            input_size=embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
            R=self.R
        )

        self.temporal_transformer_action = CP_TransformerDecoder_Action(
            input_size=embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
            R=self.R,
        )

        # 5. define codebook
        self.num_code = policy_cfg.num_code
        self.codebook = nn.parameter.Parameter(torch.randn(self.num_code, embed_size))  # 此处每个动作对应得vector维度设置为和隐层维度相同

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
        memory = self.codebook.unsqueeze(0)
        memory = repeat(memory, "b n d -> (repeat b) n d", repeat=x.shape[0])  # (B:32, num_code:100, E)
        self.temporal_transformer_skill.compute_mask(x.shape, memory.shape)
        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        x = self.temporal_transformer_skill(x, memory, CP_U=self.CP_U, CP_V=self.CP_V, CP_C=self.CP_C)
        x = x.reshape(*sh)  # (B, T:10, num_modalities:5, E:384)
        # print("selected skill shape: ", x.shape)
        return x[:, :, 0]  # (B, T, E)

    def action_execute_decode(self, x):
        # print("x: ", x.shape)  (1, 10, E)
        pos_emb = self.temporal_position_encoding_fn(x)  # (10, E)
        x = x.unsqueeze(2) + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape  # (1, 10, 1, 64)
        self.temporal_transformer_action.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        x = self.temporal_transformer_action(x, CP_U=self.CP_U, CP_V=self.CP_V, CP_C=self.CP_C)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

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
        x = self.skill_inference_decode(x)  # (B:32, T:10, E)
        # print("after skill_inference_decode shape: ", x.shape)
        x = self.action_execute_decode(x)
        # print("after action_execute_decode shape: ", x.shape)   (B, T, E)
        dist = self.policy_head(x)  # .sample()能够得到动作 [B:32, T:10, ac_dim:7]
        return dist, self.temporal_transformer_skill.attention_output, self.codebook

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
            x = self.skill_inference_decode(x)
            x = self.action_execute_decode(x)
            dist = self.policy_head(x[:, -1])  # 选取了最后一个时间步
        action = dist.sample().detach().cpu()  # 动作采样 (T:1, ac_dim:7)
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.latent_queue = []

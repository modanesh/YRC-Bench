import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, image_obs_shape, vector_obs_shape, action_shape):
        super().__init__()
        self.image_obs_shape = image_obs_shape
        self.vector_obs_shape = vector_obs_shape

        self.critic_conv = nn.Sequential(
            layer_init(nn.Conv2d(image_obs_shape[-1], 256, kernel_size=8, stride=4)),  # Note: input channels last in shape
            nn.ReLU(),
            layer_init(nn.Conv2d(256, 128, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 16, kernel_size=3, stride=1)),
            nn.ReLU()
        )

        self.actor_conv = nn.Sequential(
            layer_init(nn.Conv2d(image_obs_shape[-1], 256, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(256, 128, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 16, kernel_size=3, stride=1)),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(image_obs_shape)
        combined_input_size = conv_out_size + vector_obs_shape[0]

        self.critic_linear = nn.Sequential(
            layer_init(nn.Linear(combined_input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean_linear = nn.Sequential(
            layer_init(nn.Linear(combined_input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape[-3:])  # *shape[-3:] to pass only (C, H, W) part
        o = o.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
        o = self.critic_conv(o)
        return int(np.prod(o.size()))

    def forward_conv(self, x, conv):
        x = x.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
        return conv(x).reshape(x.size(0), -1)

    def get_value(self, img_x, vct_x):
        img_x, vct_x = self.dim_fixer(img_x, vct_x)
        conv_out = self.forward_conv(img_x, self.critic_conv)
        combined_input = torch.cat([conv_out, vct_x], dim=1)
        return self.critic_linear(combined_input)

    def get_action_and_value(self, img_x, vct_x, action=None):
        img_x, vct_x = self.dim_fixer(img_x, vct_x)
        conv_out = self.forward_conv(img_x, self.actor_conv)
        combined_input = torch.cat([conv_out, vct_x], dim=1)
        action_mean = self.actor_mean_linear(combined_input)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(img_x, vct_x)

    def dim_fixer(self, img_x, vct_x):
        if len(img_x.shape) == 3:
            img_x = img_x.unsqueeze(0)
            vct_x = vct_x.unsqueeze(0)
        return img_x, vct_x


# import robomimic.utils.tensor_utils as TensorUtils
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# import numpy as np
#
#
# from libero.lifelong.models.modules.transformer_modules import TransformerDecoder
# from libero.lifelong.models.modules.data_augmentation import DataAugGroup
# from libero.lifelong.models.bc_transformer_policy import ExtraModalityTokens
#
#
# REGISTERED_POLICIES = {}
#
#
# def register_policy(policy_class):
#     """Register a policy class with the registry."""
#     policy_name = policy_class.__name__.lower()
#     if policy_name in REGISTERED_POLICIES:
#         raise ValueError("Cannot register duplicate policy ({})".format(policy_name))
#
#     REGISTERED_POLICIES[policy_name] = policy_class
#
#
# def get_policy_class(policy_name):
#     """Get the policy class from the registry."""
#     if policy_name.lower() not in REGISTERED_POLICIES:
#         raise ValueError(
#             "Policy class with name {} not found in registry".format(policy_name)
#         )
#     return REGISTERED_POLICIES[policy_name.lower()]
#
#
# def get_policy_list():
#     return REGISTERED_POLICIES
#
#
# class PolicyMeta(type):
#     """Metaclass for registering environments"""
#
#     def __new__(meta, name, bases, class_dict):
#         cls = super().__new__(meta, name, bases, class_dict)
#
#         # List all policies that should not be registered here.
#         _unregistered_policies = ["BasePolicy"]
#
#         if cls.__name__ not in _unregistered_policies:
#             register_policy(cls)
#         return cls
#
#
# class BasePolicy(nn.Module, metaclass=PolicyMeta):
#     def __init__(self, cfg, shape_meta):
#         super().__init__()
#         self.cfg = cfg
#         self.device = cfg.device
#         self.shape_meta = shape_meta
#
#         policy_cfg = cfg.policy
#
#         # add data augmentation for rgb inputs
#         color_aug = eval(policy_cfg.color_aug.network)(
#             **policy_cfg.color_aug.network_kwargs
#         )
#
#         policy_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta[
#             "all_shapes"
#         ][cfg.data.obs.modality.rgb[0]]
#         translation_aug = eval(policy_cfg.translation_aug.network)(
#             **policy_cfg.translation_aug.network_kwargs
#         )
#         self.img_aug = DataAugGroup((color_aug, translation_aug))
#
#     def forward(self, data):
#         """
#         The forward function for training.
#         """
#         raise NotImplementedError
#
#     def get_action(self, data):
#         """
#         The api to get policy's action.
#         """
#         raise NotImplementedError
#
#     def _get_img_tuple(self, data):
#         img_tuple = tuple(
#             [data["obs"][img_name] for img_name in self.image_encoders.keys()]
#         )
#         return img_tuple
#
#     def _get_aug_output_dict(self, out):
#         img_dict = {
#             img_name: out[idx]
#             for idx, img_name in enumerate(self.image_encoders.keys())
#         }
#         return img_dict
#
#     def preprocess_input(self, data, train_mode=True):
#         if train_mode:  # apply augmentation
#             if self.cfg.train.use_augmentation:
#                 img_tuple = self._get_img_tuple(data)
#                 aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
#                 for img_name in self.image_encoders.keys():
#                     data["obs"][img_name] = aug_out[img_name]
#             return data
#         else:
#             data = TensorUtils.recursive_dict_list_tuple_apply(
#                 data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # add time dimension
#             )
#             data["task_emb"] = data["task_emb"].squeeze(1)
#         return data
#
#     def compute_loss(self, data, reduction="mean"):
#         data = self.preprocess_input(data, train_mode=True)
#         dist = self.forward(data)
#         loss = self.policy_head.loss_fn(dist, data["actions"], reduction)
#         return loss
#
#     def reset(self):
#         """
#         Clear all "history" of the policy if there exists any.
#         """
#         pass
#
#
# class PPOTransformerPolicy(BasePolicy):
#     """
#     Input: (o_{t-H}, ... , o_t)
#     Output: a_t or distribution of a_t
#     """
#
#     def __init__(self, cfg, shape_meta):
#         super().__init__(cfg, shape_meta)
#         policy_cfg = cfg.policy
#
#         ### 1. encode image
#         embed_size = policy_cfg.embed_size
#         transformer_input_sizes = []
#         self.image_encoders = {}
#         for name in shape_meta["all_shapes"].keys():
#             if "rgb" in name or "depth" in name:
#                 kwargs = policy_cfg.image_encoder.network_kwargs
#                 kwargs.input_shape = shape_meta["all_shapes"][name]
#                 kwargs.output_size = embed_size
#                 kwargs.language_dim = (
#                     policy_cfg.language_encoder.network_kwargs.input_size
#                 )
#                 self.image_encoders[name] = {
#                     "input_shape": shape_meta["all_shapes"][name],
#                     "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
#                 }
#
#         self.encoders = nn.ModuleList(
#             [x["encoder"] for x in self.image_encoders.values()]
#         )
#
#         ### 2. encode language
#         policy_cfg.language_encoder.network_kwargs.output_size = embed_size
#         self.language_encoder = eval(policy_cfg.language_encoder.network)(
#             **policy_cfg.language_encoder.network_kwargs
#         )
#
#         ### 3. encode extra information (e.g. gripper, joint_state)
#         self.extra_encoder = ExtraModalityTokens(
#             use_joint=cfg.data.use_joint,
#             use_gripper=cfg.data.use_gripper,
#             use_ee=cfg.data.use_ee,
#             extra_num_layers=policy_cfg.extra_num_layers,
#             extra_hidden_size=policy_cfg.extra_hidden_size,
#             extra_embedding_size=embed_size,
#         )
#
#         ### 4. define temporal transformer
#         policy_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
#         self.temporal_position_encoding_fn = eval(
#             policy_cfg.temporal_position_encoding.network
#         )(**policy_cfg.temporal_position_encoding.network_kwargs)
#
#         self.temporal_transformer = TransformerDecoder(
#             input_size=embed_size,
#             num_layers=policy_cfg.transformer_num_layers,
#             num_heads=policy_cfg.transformer_num_heads,
#             head_output_size=policy_cfg.transformer_head_output_size,
#             mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
#             dropout=policy_cfg.transformer_dropout,
#         )
#
#         policy_head_kwargs = policy_cfg.policy_head.network_kwargs
#         policy_head_kwargs.input_size = embed_size
#         policy_head_kwargs.output_size = shape_meta["ac_dim"]
#
#         self.policy_head = eval(policy_cfg.policy_head.network)(
#             **policy_cfg.policy_head.loss_kwargs,
#             **policy_cfg.policy_head.network_kwargs
#         )
#
#         self.latent_queue = []
#         self.max_seq_len = policy_cfg.transformer_max_seq_len
#
#     def temporal_encode(self, x):
#         pos_emb = self.temporal_position_encoding_fn(x)
#         x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
#         sh = x.shape
#         self.temporal_transformer.compute_mask(x.shape)
#
#         x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
#         x = self.temporal_transformer(x)
#         x = x.reshape(*sh)
#         return x[:, :, 0]  # (B, T, E)
#
#     def spatial_encode(self, data):
#         # 1. encode extra
#         extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E)
#
#         # 2. encode language, treat it as action token
#         B, T = extra.shape[:2]
#         text_encoded = self.language_encoder(data)  # (B, E)
#         text_encoded = text_encoded.view(B, 1, 1, -1).expand(
#             -1, T, -1, -1
#         )  # (B, T, 1, E)
#         encoded = [text_encoded, extra]
#
#         # 3. encode image
#         for img_name in self.image_encoders.keys():
#             x = data["obs"][img_name]
#             B, T, C, H, W = x.shape
#             img_encoded = self.image_encoders[img_name]["encoder"](
#                 x.reshape(B * T, C, H, W),
#                 langs=data["task_emb"]
#                 .reshape(B, 1, -1)
#                 .repeat(1, T, 1)
#                 .reshape(B * T, -1),
#             ).view(B, T, 1, -1)
#             encoded.append(img_encoded)
#         encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
#         return encoded
#
#     def forward(self, data):
#         x = self.spatial_encode(data)
#         x = self.temporal_encode(x)
#         dist = self.policy_head(x)
#         return dist
#
#     def get_action(self, data):
#         self.eval()
#         with torch.no_grad():
#             data = self.preprocess_input(data, train_mode=False)
#             x = self.spatial_encode(data)
#             self.latent_queue.append(x)
#             if len(self.latent_queue) > self.max_seq_len:
#                 self.latent_queue.pop(0)
#             x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
#             x = self.temporal_encode(x)
#             dist = self.policy_head(x[:, -1])
#         action = dist.sample().detach().cpu()
#         return action.view(action.shape[0], -1).numpy()
#
#     def reset(self):
#         self.latent_queue = []

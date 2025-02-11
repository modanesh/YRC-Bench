import numpy as np
import torch
import torch.nn.functional as F

from cliport.models.core.attention import Attention
import cliport.models as models
import cliport.models.core.fusion as fusion


class TwoStreamAttentionLangFusion(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type'] if 'train' in cfg else 'add'
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        if isinstance(inp_img, np.ndarray):
            in_data = np.pad(inp_img, self.padding, mode='constant')
            in_shape = (1,) + in_data.shape
            in_data = in_data.reshape(in_shape)
            in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]
        else:
            pad = tuple(self.padding[::-1].flatten())
            in_data = F.pad(inp_img, pad, mode='constant')
            in_shape = (1,) + in_data.shape
            in_data = in_data.reshape(in_shape)
            in_tens = in_data

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        features = []
        for x in in_tens:
            lgts, ftrs = self.attend(x, lang_goal)
            logits.append(lgts)
            features.append(ftrs)
        logits = torch.cat(logits, dim=0)
        features = torch.cat(features, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output, features


class TwoStreamAttentionLangFusionLat(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type'] if 'train' in cfg else 'add'
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def attend(self, x, l):
        x1, lat, st_one_features = self.attn_stream_one(x)
        x2, st_two_features = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)

        features = torch.cat((st_one_features.flatten(), st_two_features.flatten()))
        return x, features

    def get_logits(self, inp_img, lang_goal):
        """
        Return a single logit for the most confident pick location.

        This function processes the input image and language goal to compute
        a reduced logit representation focused on the most confident pick location.
        It simplifies the output by retaining only the confidence value corresponding
        to the highest pick location likelihood.

        Args:
            inp_img (np.ndarray or torch.Tensor): Input image (RGB-D) as a numpy array or tensor.
            lang_goal (str): Language description of the goal.

        Returns:
            np.ndarray: A single-element array containing the logit value for the
                        most confident pick location.
        """
        if isinstance(inp_img, np.ndarray):
            in_data = np.pad(inp_img, self.padding, mode='constant')
            in_shape = (1,) + in_data.shape
            in_data = in_data.reshape(in_shape)
            in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]
        else:
            pad = tuple(self.padding[::-1].flatten())
            in_data = F.pad(inp_img, pad, mode='constant')
            in_shape = (1,) + in_data.shape
            in_data = in_data.reshape(in_shape)
            in_tens = in_data

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        features = []
        for x in in_tens:
            lgts, ftrs = self.attend(x, lang_goal)
            logits.append(lgts)
            features.append(ftrs)
        logits = torch.cat(logits, dim=0)
        features = torch.cat(features, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        logits = logits.detach().cpu().numpy()
        max_idx = np.unravel_index(np.argmax(logits), logits.shape)
        return torch.tensor(logits[max_idx].reshape(1)).to(self.device)

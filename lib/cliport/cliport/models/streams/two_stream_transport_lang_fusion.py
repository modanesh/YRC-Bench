import torch
import numpy as np
import torch.nn.functional as F

import cliport.models as models
import cliport.models.core.fusion as fusion
from cliport.models.core.transport import Transport


class TwoStreamTransportLangFusion(Transport):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type'] if 'train' in cfg else 'conv'
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l):
        logits = self.fusion_key(self.key_stream_one(in_tensor), self.key_stream_two(in_tensor, l))
        kernel = self.fusion_query(self.query_stream_one(crop), self.query_stream_two(crop, l))
        return logits, kernel

    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        if isinstance(inp_img, np.ndarray):
            in_data = np.pad(inp_img, self.padding, mode='constant')
            in_shape = (1,) + in_data.shape
            in_data = in_data.reshape(in_shape)
            in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)
        else:
            pad = tuple(self.padding[::-1].flatten())
            in_data = F.pad(inp_img, pad, mode='constant')
            in_shape = (1,) + in_data.shape
            in_data = in_data.reshape(in_shape)
            in_tens = in_data

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tens = in_tens.permute(0, 3, 1, 2)

        crop = in_tens.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0] - hcrop:pv[0] + hcrop, pv[1] - hcrop:pv[1] + hcrop]

        logits, kernel, features = self.transport(in_tens, crop, lang_goal)

        # TODO(Mohit): Crop after network. Broken for now.
        # # Crop after network (for receptive field, and more elegant).
        # in_tens = in_tens.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tens, lang_goal)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)
        # hcrop = self.pad_size
        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        return self.correlate(logits, kernel, softmax), features


class TwoStreamTransportLangFusionLat(TwoStreamTransportLangFusion):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type'] if 'train' in cfg else 'conv'
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one, st_one_features = self.key_stream_one(in_tensor)
        key_out_two, st_two_features = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one, _ = self.query_stream_one(crop)
        query_out_two, _ = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)
        features = torch.cat((st_one_features.flatten(), st_two_features.flatten()))
        return logits, kernel, features

    def get_logits(self, inp_img, lang_goal):
        """
        Return logits for the most confident spatial location, keeping rotation logits for 3 bins.

        Args:
            inp_img (np.ndarray or torch.Tensor): Input image (RGB-D) as a numpy array or tensor.
            lang_goal (str): Language description of the goal.

        Returns:
            np.ndarray: An array of size 3 containing logits for rotation bins at the most confident
                        spatial location.
        """
        if isinstance(inp_img, np.ndarray):
            in_data = np.pad(inp_img, self.padding, mode='constant')
            in_shape = (1,) + in_data.shape
            in_data = in_data.reshape(in_shape)
            in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)
        else:
            pad = tuple(self.padding[::-1].flatten())
            in_data = F.pad(inp_img, pad, mode='constant')
            in_shape = (1,) + in_data.shape
            in_data = in_data.reshape(in_shape)
            in_tens = in_data

        in_tens = in_tens.permute(0, 3, 1, 2)
        key_out_one, key_lat_one, st_one_features = self.key_stream_one(in_tens)
        key_out_two, st_two_features = self.key_stream_two(in_tens, key_lat_one, lang_goal)
        logits = self.fusion_key(key_out_one, key_out_two)
        logits = logits.detach().cpu().numpy()
        spatial_conf = logits[0].sum(axis=0)  # Sum over rotation channels for spatial confidence (shape: 384x224).
        max_spatial_idx = np.unravel_index(np.argmax(spatial_conf), spatial_conf.shape)  # (height, width)
        # Extract rotation logits for the most confident spatial location.
        rotation_logits = logits[0, :, max_spatial_idx[0], max_spatial_idx[1]]  # Retain 3 rotation logits.
        return torch.tensor(rotation_logits).to(self.device)

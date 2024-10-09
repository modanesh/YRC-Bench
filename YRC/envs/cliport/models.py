from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetLatTransporterAgent
import torch
from YRC.core.configs.global_configs import get_global_variable
import numpy as np


class CliportModel(TwoStreamClipLingUNetLatTransporterAgent):
    def __init__(self, name, cfg):
        super().__init__(name, cfg, train_ds=None, test_ds=None)

        dummy_image = np.ones((1, *self.in_shape), dtype=np.float32)
        dummy_info = [{"lang_goal": 'put the red block on the lightest brown block'}]
        self.to(get_global_variable("device"))
        dummy_pick_features, dummy_place_features = self.extract_features(dummy_image, dummy_info)
        self.hidden_dim = torch.cat((dummy_pick_features[0], dummy_place_features[0]), dim=0).shape[0]
        self.logit_dim = self.get_logit({"img": dummy_image[0], "info": dummy_info[0]}).shape[0]

    def get_hidden(self, obs):
        img, info = obs["img"], obs["info"]
        pick_hidden, place_hidden = super().extract_features([img], [info])
        pick_hidden = torch.stack(pick_hidden) if isinstance(pick_hidden, list) else pick_hidden
        place_hidden = torch.stack(place_hidden) if isinstance(place_hidden, list) else place_hidden

        if pick_hidden.dim() != 2:
            pick_hidden = pick_hidden.unsqueeze(0)
            place_hidden = place_hidden.unsqueeze(0)

        return torch.cat([pick_hidden, place_hidden], dim=-1)

    def get_logit(self, obs):
        img, lang_goal = obs["img"], obs["info"]['lang_goal']
        attention_logits = self.attention.get_logits(img, lang_goal)
        transport_logits = self.transport.get_logits(img, lang_goal)
        return torch.cat([attention_logits.flatten(), transport_logits.flatten()], dim=0)
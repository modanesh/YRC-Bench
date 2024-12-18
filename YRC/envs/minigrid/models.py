import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from YRC.core.configs.global_configs import get_global_variable
from YRC.models.utils import init_params


class MinigridModel(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.device = get_global_variable("device")
        obs_space, action_space = env.observation_space.spaces, env.action_space

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.word_embedding_size = 32
        self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
        self.text_embedding_size = 128
        self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.hidden_dim = self.semi_memory_size + self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)
        self.logit_dim = env.action_space.n

    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)
        embed_text = self._get_embed_text(obs.text)
        embedding = torch.cat((embedding, embed_text), dim=1)
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value

    def get_hidden(self, obs):
        """Extract the hidden embedding for the observation."""
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)
        embed_text = self._get_embed_text(obs.text)
        embedding = torch.cat((embedding, embed_text), dim=1)
        return embedding

    def get_logit(self, obs):
        """Compute the logits for the given observation."""
        embedding, _ = self.get_hidden(obs)
        logit = self.actor(embedding)
        return logit

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

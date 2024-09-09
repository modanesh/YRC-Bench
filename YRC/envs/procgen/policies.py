from torch.distributions.categorical import Categorical
from YRC.core.policy import Policy


class ProcgenPolicy(Policy):

    def __init__(self, model):
        self.model = model
        #self.temp = 1

    def forward(self, obs):
        return self.model.get_logit(obs)

    def predict(self, obs):
        dist, value = self.model(obs)
        return dist

    def act(self, obs, greedy=False, mask=None):
        #logit = self.forward(obs)
        #dist = Categorical(logits=logit / self.temp)
        dist = self.predict(obs)
        if greedy:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.cpu().numpy()

    def get_hidden(self, obs):
        return self.model.get_hidden(obs)

    @property
    def hidden_dim(self):
        return self.model.hidden_dim

    #def set_temp(self, temp):
    #    self.temp = temp

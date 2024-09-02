from YRC.core.policy import Policy


class ProcgenPolicy(Policy):

    def __init__(self, model):
        self.model = model

    def predict(self, obs):
        dist, value = self.model(obs)
        return dist

    def act(self, obs):
        a = self.predict(obs).probs.argmax(dim=-1)
        return a.cpu().numpy()

    def get_hidden_features(self, obs):
        return self.model.get_hidden(obs)

    @property
    def hidden_size(self):
        return self.model.hidden_dim

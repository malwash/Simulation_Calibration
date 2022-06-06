from dg_models.DGModel import DGModel
from pomegranate.BayesianNetwork import BayesianNetwork
from utils.Data import Data


class PomegranateLearner(DGModel):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, SLClass=BayesianNetwork, **kwargs)

    def instantiate(self):
        self.model = self.SLClass()

    def fit(self, data, algorithmstr="greedy"):
        self.model = self.model.from_samples(X=data, state_names=data.columns.values)
        self.model = self.model.fit(X=data)

    def _generate(self, num_samples, outcome_name: str, world: str = "real", dataset: str = "train"):
        data = self.model.sample(n=num_samples)
        data = Data(name=world + self.name + dataset, X=data.drop(outcome_name, axis=1),
                    y=data.loc[:, outcome_name])
        return data

from dg_models.DGModel import DGModel
import pandas as pd
from dagsim.base import Graph
from utils.Data import Data


class DagsimModel(DGModel):
    def __init__(self, name, dagsim_model: Graph):
        super().__init__(name=name, SLClass=None, learned=True)
        self.model = dagsim_model

    def instantiate(self):
        pass

    def fit(self, data: pd.DataFrame, **kwargs):
        pass

    def _generate(self, num_samples: int, outcome_name: str):
        data = pd.DataFrame.from_dict(self.model.simulate(num_samples))
        data = Data(name=self.name, X=data.drop(outcome_name, axis=1),
                    y=data.loc[:, outcome_name])
        return data

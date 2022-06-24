import pandas as pd

from dg_models.DGModel import DGModel
from utils.Data import Data
from notears.linear import notears_linear
from notears.nonlinear import notears_nonlinear, NotearsMLP
from notears import utils


class NotearsLearner(DGModel):
    def __init__(self, name, SLClass: str, **kwargs):
        super().__init__(name=name, SLClass=SLClass, **kwargs)
        self.var_names = None
        self.outcome_name = None

    def instantiate(self):
        pass

    def fit(self, data, lambda1=0.01, loss_type='logistic', dims=(10, 2)):
        self.var_names = data.X.columns.to_list()
        self.num_vars = len(self.var_names) + 1
        self.outcome_name = data.y.name
        data = data.all.to_numpy()
        if self.SLClass == "linear":
            self.model = notears_linear(X=data, lambda1=lambda1, loss_type=loss_type)
            self.learned = True
        elif self.SLClass == "nonlinear":
            raise NotImplementedError
            # self.model = notears_nonlinear(X=data)
        else:
            raise TypeError(f'Type "{self.kwargs["type"]}" is not defined')

    def _generate(self, num_samples, outcome_name: str = "Y", sem_type='logistic'):
        data = utils.simulate_linear_sem(W=self.model, n=num_samples, sem_type=sem_type)
        X = pd.DataFrame(data[:, :-1], columns=self.var_names)
        y = pd.Series(data[:, -1], name=self.outcome_name)
        data = Data(name=self.name, X=X, y=y)
        return data


if __name__ == "__main__":
    pass

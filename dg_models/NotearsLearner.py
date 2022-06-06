from DGModel import DGModel
from notears.linear import notears_linear
from notears.nonlinear import notears_nonlinear, NotearsMLP
from notears import utils


class NotearsLearner(DGModel):
    def __init__(self, name, SLClass: str, **kwargs):
        super().__init__(name=name, SLClass=SLClass, **kwargs)

    def instantiate(self):
        pass

    def fit(self, data, lambda1=0.01, loss_type='logistic', dims=(10, 2)):
        # todo: check dims
        data = data.to_numpy()
        if self.SLClass == "linear":
            self.model = notears_linear(X=data, lambda1=lambda1, loss_type=loss_type)
        elif self.SLClass == "nonlinear":
            raise NotImplementedError
            # self.model = notears_nonlinear(X=data)
        else:
            raise TypeError(f'Type "{self.kwargs["type"]}" is not defined')

    def _generate(self, num_samples, sem_type='logistic'):
        return utils.simulate_linear_sem(self.model, num_samples, sem_type)


if __name__ == "__main__":
    pass

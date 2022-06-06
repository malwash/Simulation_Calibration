import bnlearn
from notears import utils
from utils.notears_class import Notears
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator


class WorldLearner:
    def __init__(self, model_class, **kwargs):
        self.model = None
        self.model_class = model_class
        self.package = self.get_package_name()
        self.kwargs = kwargs

    def get_package_name(self):
        module_name = self.model_class.__module__.split(".")[0]
        return module_name

    def learn(self, train_data):
        if self.package == "pgmpy":
            init_model = self.model_class(train_data)
            init_model = init_model.estimate()
            construct = BayesianModel(init_model)
            estimator = BayesianEstimator(construct, train_data)
            cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=1000)
            for cpd in cpds:
                construct.add_cpds(cpd)
            construct.check_model()
            self.model = construct
        elif self.package == "pomegranate":
            self.model = self.model_class()
            self.model.from_samples(train_data)
        elif self.package == "notears":
            self.model = Notears(**self.kwargs)
            self.model.fit(train_data)

    def generate(self, num_samples):
        if self.package == "pgmpy":
            return self.model.simulate(num_samples)
        elif self.package == "pomegranate":
            return self.model.sample(num_samples)
        elif self.package == "notears":
            return self.model.simulate(num_samples)

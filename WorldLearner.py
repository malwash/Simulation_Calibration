import bnlearn
import notears
import importlib

class WorldLearner:
    def __init__(self, pkg_name, alg_name):
        self.pkg_name = pkg_name
        self.alg_name = alg_name
        self.learner = self.get_alg(pkg_name, alg_name)
        self.model = None

    def get_alg(self, pkg_name, alg_name):
        # For functions not defined in the python file, but rather coming from external libraries, such as numpy.
        module = importlib.import_module(pkg_name)
        function = getattr(module, alg_name)
        return function

    def learn(self, train_data, test_data):
        self.model = self.learner(train_data, test_data)

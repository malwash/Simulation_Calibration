import unittest
from dg_models.PomegranateLearner import PomegranateLearner
import pandas as pd
import numpy as np


class PomegranateModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = pd.DataFrame(np.random.randint(0, 2, size=(20, 2)), columns=list('AB'))

    def test_instantiate(self):
        try:
            model = PomegranateLearner(name="test")
        except Exception as exc:
            assert False, f"pomegranate instantiate raised an exception: {exc}"

    def test_fit(self):
        try:
            model = PomegranateLearner(name="test")
            model.fit(self.data)
        except Exception as exc:
            assert False, f"pomegranate fit raised an exception: {exc}"

    def test_generate(self):
        try:
            model = PomegranateLearner(name="test")
            model.fit(self.data)
            data = model.generate(5)
        except Exception as exc:
            assert False, f"pomegranate generate raised an exception: {exc}"


if __name__ == '__main__':
    unittest.main()

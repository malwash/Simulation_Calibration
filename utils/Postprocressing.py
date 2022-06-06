import matplotlib.pyplot as plt
import pandas as pd
from transpose_dict import TD


class Postprocessing():
    def __init__(self, results: dict):
        self.results = results
        sample_key = list(results.keys())[0]
        self.ml_algs = list(TD(results[sample_key], 1).keys())
        self.worlds = list(results[sample_key].keys())

    def plot_scores(self, pipeline):
        data = self.dict_to_list(pipeline)
        df = pd.DataFrame(data, columns=["ML", *self.worlds])

        ax = df.plot(x="ML", y=self.worlds, kind="bar", figsize=(9, 8))
        ax.set_ylim(0, 1)
        plt.show()

    def dict_to_list(self, pipeline):
        data = []
        for alg in self.ml_algs:
            inner_list = [alg]
            for world in self.worlds:
                inner_list.append(self.results[pipeline][world][alg]["accuracy"])
            data.append(inner_list)
        return data

    def dict_to_dataframe(self, pipeline):
        emdf = pd.DataFrame(index=self.ml_algs, columns=self.worlds)
        for ml_alg in self.ml_algs:
            for world in self.worlds:
                emdf.at[ml_alg, world] = self.results[pipeline][world][ml_alg]["accuracy"]
        return emdf


if __name__ == "__main__":
    results = {"pipeline1": {
        'real_model': {'LogisticRegression_alg': {'accuracy': 0.925}, 'BernoulliNB_alg': {'accuracy': 0.925},
                       'KNeighborsClassifier_alg': {'accuracy': 0.945}},
        'HillClimbSearch_alg': {'LogisticRegression_alg': {'accuracy': 0.83},
                                'BernoulliNB_alg': {'accuracy': 0.83},
                                'KNeighborsClassifier_alg': {'accuracy': 0.83}},
        'TreeSearch_alg': {'LogisticRegression_alg': {'accuracy': 0.865},
                           'BernoulliNB_alg': {'accuracy': 0.865},
                           'KNeighborsClassifier_alg': {'accuracy': 0.865}}}}

    pp = Postprocessing(results)
    df = pp.dict_to_dataframe("pipeline1")
    ll = pp.dict_to_list("pipeline1")
    pp.plot_scores("pipeline1")

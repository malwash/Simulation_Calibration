import matplotlib.pyplot as plt
import pandas as pd
from transpose_dict import TD
import numpy as np
import seaborn as sns
from typing import List
from dg_models.DGModel import DGModel


class Postprocessing():
    def __init__(self):
        pass

    # todo give better name for different plotting functions
    def plot_scores(self, pipeline: str, all_results: dict):
        for score_name in TD(all_results, 3).keys():
            data = self.dict_to_list(pipeline, score_name, all_results)
            # todo chage way to get world names
            sample_key = list(all_results.keys())[0]
            worlds = list(all_results[sample_key].keys())
            df = pd.DataFrame(data, columns=["ML", *worlds])

            ax = df.plot(x="ML", y=worlds, kind="bar", figsize=(9, 8))
            ax.set_ylim(0, 1)
            ax.set_ylabel(score_name)
            plt.show()

    def dict_to_list(self, pipeline, score_name, all_results: dict):
        data = []
        sample_key = list(all_results.keys())[0]
        ml_algs = list(TD(all_results[sample_key], 1).keys())
        worlds = list(all_results[sample_key].keys())
        for alg in ml_algs:
            inner_list = [alg]
            for world in worlds:
                inner_list.append(all_results[pipeline][world][alg][score_name])
            data.append(inner_list)
        return data

    def dict_to_dataframe(self, pipeline):
        emdf = pd.DataFrame(index=self.ml_algs, columns=self.worlds)
        for ml_alg in self.ml_algs:
            for world in self.worlds:
                emdf.at[ml_alg, world] = self.all_results[pipeline][world][ml_alg]["accuracy_score"]
        return emdf

    def plot_analysis1(self, analysis1_results: pd.DataFrame):
        score_names = analysis1_results.index
        for score_name in score_names:
            y = [np.mean(analysis1_results[alg][score_name]) for alg in analysis1_results.columns]
            y_err_d = [np.mean(analysis1_results[alg][score_name]) - np.min(analysis1_results[alg][score_name]) for alg in analysis1_results.columns]
            y_err_u = [np.max(analysis1_results[alg][score_name]) - np.mean(analysis1_results[alg][score_name]) for alg in analysis1_results.columns]
            y_err = [y_err_d, y_err_u]

            alg_names = analysis1_results.columns
            x_pos = np.arange(len(alg_names))

            fig, ax = plt.subplots()
            ax.bar(x_pos, y,
                   yerr=y_err,
                   align='center',
                   alpha=0.5,
                   ecolor='black',
                   capsize=10)
            ax.set_ylim(0, 1)
            ax.set_ylabel(score_name)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(alg_names)
            # ax.set_title(f'{score_name} of different ML models')
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig('bar_plot_with_error_bars.png')
            plt.show()

    # not used for now
    def get_true_performance_stats(self, scores: dict):
        # scores: dict of shape {"ml_model_name": {"score_name": list_of_values}}
        score_names = TD(scores, 1).keys()
        stats = {ml_model_name: {score_name: {} for score_name in score_names} for ml_model_name in scores.keys()}
        for ml_model_name in scores.keys():
            for score in score_names:
                stats[ml_model_name][score]["min"] = np.min(scores[ml_model_name][score])
                stats[ml_model_name][score]["max"] = np.max(scores[ml_model_name][score])
                stats[ml_model_name][score]["mean"] = np.mean(scores[ml_model_name][score])
        return stats

    def dict_to_dataframe_sns(self, scores: dict, score_name: str = 'balanced_accuracy_score') -> pd.DataFrame:
        '''

        :param scores: dict of form {world_name: {ml_model_name: {score_name: list of scores]...}...}...}
        :param score_name:
        :return: dataframe with columns=[world, ml_model, score] and rows corresponding to individual entries
        '''
        score_dict = TD(scores, 2)[score_name]
        raw_scores_df = pd.DataFrame(score_dict)
        list_of_lists = []

        for col in raw_scores_df.columns:
            for row in raw_scores_df.index:
                for i in range(5):
                    list_of_lists.append([col, row, raw_scores_df[col][row][i]])

        return pd.DataFrame(list_of_lists, columns=["world", "ml_model", "score"])

    def plot_analysis4(self, scores: dict, score_name='balanced_accuracy_score'):
        unfolded_scores = self.dict_to_dataframe_sns(scores, score_name)
        ax = sns.violinplot(x="world", y="score", hue="ml_model", data=unfolded_scores)
        plt.show()

    def corr_dict_to_pd(self, corr_dict):
        corr_pd = pd.concat(corr_dict)
        corr_pd.rename(columns={0: 'correlation'}, inplace=True)
        corr_pd.index = corr_pd.index.set_names(['world', 'pair'])
        corr_pd.reset_index(level=['world', 'pair'], inplace=True)
        corr_pd["pair"] = corr_pd["level_0"] + corr_pd["level_1"]
        corr_pd.drop(columns=["level_0", "level_1"])
        return corr_pd

    def plot_correlations(self, corr_dict: dict):
        corr_df = self.corr_dict_to_pd(corr_dict)
        sns_plot = sns.scatterplot(x='pair', y='correlation', data=corr_df, hue='world', alpha=0.6, style="world")
        # sns_plot.set(yscale="log")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()


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

    pp = Postprocessing()
    # df = pp.dict_to_dataframe("pipeline1")
    # ll = pp.dict_to_list("pipeline1")
    pp.plot_scores("pipeline1")

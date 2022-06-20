import matplotlib.pyplot as plt
import pandas as pd
from transpose_dict import TD
import numpy as np
import seaborn as sns


class Postprocessing():
    def __init__(self):
        pass

    def plot_analysis1(self, analysis1_results: pd.DataFrame):
        score_names = analysis1_results.index
        for score_name in score_names:
            y = [np.mean(analysis1_results[alg][score_name]) for alg in analysis1_results.columns]
            y_err_d = [np.mean(analysis1_results[alg][score_name]) - np.min(analysis1_results[alg][score_name]) for alg
                       in analysis1_results.columns]
            y_err_u = [np.max(analysis1_results[alg][score_name]) - np.mean(analysis1_results[alg][score_name]) for alg
                       in analysis1_results.columns]
            y_err = [y_err_d, y_err_u]

            alg_names = analysis1_results.columns
            x_pos = np.arange(len(alg_names))

            fig, ax = plt.subplots()
            ax.bar(x_pos, y, yerr=y_err, align='center', alpha=0.5, ecolor='black', capsize=10)
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

    def plot_analysis2(self, analysis2_results: dict):
        corr_df = self.corr_dict_to_pd(analysis2_results)
        sns_plot = sns.scatterplot(x='pair', y='correlation', data=corr_df, hue='world', alpha=0.6, style="world")
        plt.show()

    def plot_analysis2_gks(self, analysis2_results: dict, pipeline: str = "pipeline1"):
        markers = ['x', 'o', '^', '*', '+', 's', 'p']
        c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        corr_df = self.corr_dict_to_pd(analysis2_results)
        real_corr = corr_df.loc[corr_df['world'] == pipeline].correlation
        for i, world in enumerate(set(corr_df.world)):
            if world == "PC":
                continue
            world_corr = corr_df.loc[corr_df['world'] == world].correlation

            plt.scatter(real_corr.tolist(), world_corr.tolist(), marker=markers[i], c=c[i], label=world)

        plt.title("pair-wise correlations of learned vs real world")
        plt.xlabel("real world correlations")
        plt.ylabel("learned world correlations")
        plt.legend()
        plt.show()

    def plot_analysis3(self, pipeline: str, analysis3_results: dict):
        for score_name in TD(analysis3_results, 3).keys():
            data = self.dict_to_list(pipeline, score_name, analysis3_results)
            # todo chage way to get world names
            sample_key = list(analysis3_results.keys())[0]
            worlds = list(analysis3_results[sample_key].keys())
            df = pd.DataFrame(data, columns=["ML", *worlds])

            ax = df.plot(x="ML", y=worlds, kind="bar", figsize=(9, 8))
            ax.set_ylim(0, 1)
            ax.set_ylabel(score_name)
            plt.show()

    def plot_analysis4(self, analysis4_results: dict, score_name='balanced_accuracy_score'):
        unfolded_scores = self.dict_to_dataframe_sns(analysis4_results, score_name)
        ax = sns.violinplot(x="world", y="score", hue="ml_model", data=unfolded_scores)
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

    def corr_dict_to_pd(self, corr_dict):
        corr_pd = pd.concat(corr_dict)
        corr_pd.rename(columns={0: 'correlation'}, inplace=True)
        corr_pd.index = corr_pd.index.set_names(['world', 'pair'])
        corr_pd.reset_index(level=['world', 'pair'], inplace=True)
        corr_pd["pair"] = corr_pd["level_0"] + corr_pd["level_1"]
        corr_pd.drop(columns=["level_0", "level_1"], inplace=True)
        return corr_pd

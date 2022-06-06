import pandas as pd


class Data():
    def __init__(self, name: str, X: pd.DataFrame, y: pd.Series):
        self.name = name
        self.X = X
        self.y = y
        self.all = pd.merge(X, y, right_index=True, left_index=True)

    def sort(self):
        names = self.X

    def __getitem__(self, item):
        #  item is a slice object of three ints
        X = self.X.iloc[item]
        y = self.y.iloc[item]
        # todo fix name
        return Data(f"{self.name}_learning", X, y)

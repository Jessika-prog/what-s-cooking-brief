from imports import *


class Data:

    def __init__(self):

        self.df_train = pd.read_json('train.json')
        self.df_test = pd.read_json('test.json')
        return None

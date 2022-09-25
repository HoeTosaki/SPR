import numpy as np

from catboost import CatBoostClassifier, Pool
from catboost import CatBoostRegressor

def eval_CatBoost():
    # initialize data
    train_data = np.random.randint(0,100,size=(100, 16))
    train_labels = np.random.rand(100)
    param_grid = {'iterations':[500],'learning_rate': [0.03, 0.1], 'depth': [4, 6, 10], 'l2_leaf_reg': [3, 7]}
    model = CatBoostRegressor()

    model.grid_search(param_grid,X=train_data,y=train_labels,plot=True)


if __name__ == '__main__':
    eval_CatBoost()

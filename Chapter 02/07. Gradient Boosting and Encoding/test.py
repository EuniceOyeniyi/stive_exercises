import numpy as np
import pandas as pd
from hmac import new
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
import data_handler as dh
import train as TR
from xgboost import XGBRegressor




tree = TR.treemodel()
tree.datainspection()
# print(tree.accuracy())
# data_input = [30,	'female', 29.1,	1,	'yes',	'southwest']
# new_data = pd.DataFrame([data_input],columns=['a', 'b', 'c','d','e','f'])
# tanformer = TR.ct
# final_data = tanformer.transform(new_data)
# print(tree.prediction(final_data))

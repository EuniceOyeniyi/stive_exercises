from inspect import Parameter
import data_handler as dh
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

x_train, x_test, y_train, y_test,ct,data= dh.get_data("insurance.csv")


models = [RandomForestRegressor(random_state=0),AdaBoostRegressor(random_state=0),GradientBoostingRegressor(random_state=0),XGBRegressor(random_state=0)]

class treemodel:


    def __init__(self):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.ct = ct
        self.model = models
        self.data = data

    def datainspection(self):
        print('-----------------------------------------------------------')
        print('Checking the data information')
        print('-----------------------------------------------------------')
        print(self.data.info(),"\n")
        print('-----------------------------------------------------------')
        print('Checking the shape of the splitted data')
        print('-----------------------------------------------------------')
        print('xtrain shape is',self.x_train.shape)
        print('ytrain shape is',self.y_train.shape)
    

    def cross_validation_score(self):
        model_perfromance = []
        for model in self.model:
            clf_model = model
            performance = cross_val_score(clf_model,self.x_train, self.y_train) #evaluating the model
            model_perfromance.append(performance.mean())
        return model_perfromance


   

    def accuracy(self):
        acc_scores=[]
        for model in self.model:
            clf_model = model
            clf_model.fit(self.x_train,self.y_train)
            acuu_score = clf_model.score(self.x_test,self.y_test)
            acc_scores.append(acuu_score)
        return acc_scores


    def gridsearch(self):
        clf_model = GradientBoostingRegressor(random_state=0)
        parametes = {'n_estimators':[100,250,500],'learning_rate':np.linspace(0.001, 0.1, 5),'max_features':['auto','sqrt','log2']}
        grid_search = GridSearchCV(clf_model, parametes)
        grid_search.fit(x_train,y_train)
        tuned_model =grid_search.best_estimator_
        # print(grid_search.best_score_)
        return tuned_model


    def prediction(self,data):
        best_modelparams = self.gridsearch()
        best_modelparams.fit(self.x_train,self.y_train)
        pred = best_modelparams.predict(data)

        return pred
  


 

       

    
    





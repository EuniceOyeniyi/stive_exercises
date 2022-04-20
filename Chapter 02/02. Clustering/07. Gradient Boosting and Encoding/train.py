import data_handler as dh
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

x_train, x_test, y_train, y_test,ct = dh.get_data("insurance.csv")

# models = [RandomForestRegressor(random_state=0),AdaBoostRegressor(random_state=0),GradientBoostingRegressor(random_state=0),XGBRegressor(random_state=0)]


def cross_validation_score():
    models = [RandomForestRegressor(random_state=0),AdaBoostRegressor(random_state=0),GradientBoostingRegressor(random_state=0),XGBRegressor(random_state=0)]
    model_perfromance = []
    
    for model in models:
        clf_model = model
        performance = cross_val_score(clf_model,x_train, y_train) #evaluating the model
        model_perfromance.append(performance.mean())
    return model_perfromance



def accuracy():
    models = [RandomForestRegressor(random_state=0),AdaBoostRegressor(random_state=0),GradientBoostingRegressor(random_state=0),XGBRegressor(random_state=0)]
    acc_scores=[]
    for model in models:
        clf_model = model
        clf_model.fit(x_train,y_train)
        acuu_score = clf_model.score(x_test,y_test)
        acc_scores.append(acuu_score)
    return acc_scores




def prediction(data):
    new_data = ct.transform(data)
    clf_model = GradientBoostingRegressor(random_state=0)
    clf_model.fit(x_train,y_train)
    pred = clf_model.predict(new_data)

    return pred

    
    





import numpy as np
from hmac import new
from sklearn.ensemble import RandomForestRegressor
import train as TR

while True:

    age = int(input("How old are you? \n"))
    child = int(input("How many children do you have? \n"))
    smoke = str(input("Do you smoke? \n"))
    sex = str(input("Indicate your gender \n"))
    region= str(input("which region do you stay?option:[southeast,northeast,southwest,northwest] \n"))
    bmi = float(input("what is your BMI \n"))
    new_data = [age,sex,bmi,child,smoke,region]

    data = np.array(new_data).reshape(-6,6)
    predi =TR.prediction(data)

   
    print(predi)    

    '''
    Preprocess
    predict
    
    '''
    break


# data= new_data
# perfomance = TR.cross_validation_score()
# data = np.array([1.0, 0.0, 3.0, 52, 30.2, 1]).reshape(-6,6)
# predi =TR.prediction(data)
# print(predi)
import pandas as pd
from hmac import new
import data_handler as dh
import train as TR


while True:

    age = int(input("How old are you? \n"))
    child = int(input("How many children do you have? \n"))
    smoke = str(input("Do you smoke? \n"))
    sex = str(input("Indicate your gender \n"))
    region= str(input("which region do you stay?option:[southeast,northeast,southwest,northwest] \n"))
    bmi = float(input("what is your BMI \n"))
    data_input = [age,sex,bmi,child,smoke,region]

#     Preprocess
    new_data = pd.DataFrame([data_input],columns=['age', 'sex', 'bmi','children','smoker','region'])

    tree = TR.treemodel()
    tanformer = TR.ct
    final_data = tanformer.transform(new_data)

#     predict
    print(tree.prediction(final_data))

    break



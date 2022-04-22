import datacleaning as DC
import numpy as np
import pandas as pd



# DC.load_data_info('./data/') 

p1, p2 = DC.data_cleaning('./data/') 
print(p1.head(2))
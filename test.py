from Artificial_Neural_Network import artificialneuralnetwork_classifier
import pandas as pd
import numpy as np

if __name__ == "__main__" :
  def pre_x(x):
      x_ = np.matrix( x )
      print(x_ , "--->" , Ann.predict(x_))
      
  df = pd.read_csv('admission_data.csv')
  df = df.apply(pd.to_numeric, errors='coerce')
  df = df.dropna()
  x = np.matrix(df[["GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA"]].to_numpy() )
  y = np.matrix(df[["Research"]].to_numpy())
  Ann = artificialneuralnetwork_classifier(x,y)
  pre_x([[318,110,3,4,3,8.8] ])
  pre_x([[321,110,3,3.5,5,8.85] ])
  pre_x([[324,112,4,4,2.5,8.1] ])

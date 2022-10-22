import numpy as np
from numpy.linalg import inv
import math

class tanhiperboliqueregression:
  
  def __init__(self,training_data_X, training_data_Y) :
    
    def get_BETA() :
      pading = np.ones(self.training_data_X.shape[0])
      z = np.insert(self.training_data_X, 0, pading, axis=1)
      Beta = numpy.arctanh( ( 2 * self.training_data_Y ) - 1 ) / z
      return Beta
      
    self.training_data_X = training_data_X # The training data x => features numpy_matrix
    self.training_data_Y = training_data_Y # The training data y => response numpy_matrix
    self.Beta = get_BETA()

  def predict(self,x):
    pading = np.ones(x.shape[0])
    x = np.insert(x, 0, pading, axis=1)
    p_of_x = (0.5 * np.tanh(x * self.Beta) + 0.5 ) 
    return p_of_x.item()
    
if __name__ == "__main__" :
  x = np.matrix([[1,3],[2,4],[4,1],[3,1],[4,2] ])
  y = np.matrix([[0],[0],[0],[1],[1] ] )
  LR = tanhiperboliqueregression(x,y)
  x_ = np.matrix([[0,0]])
  print(LR.predict(x_))

import numpy as np
from numpy.linalg import inv
import math
class logisticregression:
  
  def __init__(self,training_data_X, training_data_Y) :
    
    def get_BETA() :
      pading = np.ones(self.training_data_X.shape[0])
      z = np.insert(self.training_data_X, 0, pading, axis=1)
      z_prime = z.T
      Beta_0 = np.matrix(np.zeros(z.shape[1]))
      signal = False
      while signal == False :
        P_zi = np.exp( z * Beta_0.T  ) / (1 + np.exp( z * Beta_0.T  ) )
        id = np.matrix(np.ones((P_zi.shape[0])))
        w =  np.diagflat(np.diag(( id.T - P_zi ).dot(P_zi.T)))
        v = z * Beta_0.T  + ( inv(w) *  (self.training_data_Y - P_zi ) )
        Beta_1 = (inv(z.T * w * z) * z.T * w * v).T
        if Beta_1.all() ==  Beta_0.all()  :
          signal = True
          return Beta_1.T
        else :
          Beta_0 = Beta_1
      
    self.training_data_X = training_data_X # The training data x => features numpy_matrix
    self.training_data_Y = training_data_Y # The training data y => response numpy_matrix
    self.Beta = get_BETA()

  def predict(self,x):
    pading = np.ones(x.shape[0])
    x = np.insert(x, 0, pading, axis=1)
    p_of_x = np.exp(x * self.Beta) / ( 1 + np.exp(x * self.Beta) )
    return p_of_x.item()
    
if __name__ == "__main__" :
  x = np.matrix([[1,3],[2,4],[4,1],[3,1],[4,2] ])
  y = np.matrix([[0],[0],[0],[1],[1] ] )
  LR = logisticregression(x,y)
  x_ = np.matrix([[3,7]])
  print(LR.predict(x_))

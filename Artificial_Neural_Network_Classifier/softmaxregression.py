import numpy as np
from numpy.linalg import inv
import math

class softmaxregression:
  
  def __init__(self,training_data_X, training_data_Y) :
    
    def get_BETA() :
      pading = np.ones(self.training_data_X.shape[0])
      z = np.insert(self.training_data_X, 0, pading, axis=1)
      z_prime = z.T
      Beta_0 = np.matrix(np.zeros(z.shape[1]))
      signal = False
      while signal == False :
        P_zi_Num = np.exp( -1 * z * Beta_0.T ) / ( sum(np.exp( -1 * z * Beta_0.T )) )
        P_zi = (  (P_zi_Num / (1 + P_zi_Num))  ) / (1 + ( (P_zi_Num / (1 + P_zi_Num))  ) )
        id = np.matrix(np.ones((P_zi.shape[0])))
        w =  np.diagflat(np.diag(( id.T - P_zi ).dot(P_zi.T)))
        v = z * Beta_0.T  + ( inv(w) *  (self.training_data_Y - P_zi ) )
        Beta_1 = (inv(z.T * w * z) * z.T * w * v).T
        if Beta_1.any() ==  Beta_0.any()  :
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
    p_of_x_Num = np.exp( -1 * x * self.Beta ) / ( sum( np.exp( -1 * x * self.Beta ) ) )
    p_of_x = (  (p_of_x_Num / (1 + p_of_x_Num))  ) / (1 + (  (p_of_x_Num / (1 + p_of_x_Num))  ) )
    return p_of_x.item() 
    
if __name__ == "__main__" :
  x = np.matrix([[1,3],[2,4],[4,1],[3,1],[4,2] , [5,1] ])
  y = np.matrix([[0],[0],[0],[1],[1] , [1] ] )
  LR = softmaxregression(x,y)
  x_ = np.matrix([[4,2]])
  print(LR.predict(x_))

import numpy as np
from numpy.linalg import inv

class lineardiscriminantananlysis :



  def __init__(self,training_data_X, training_data_Y) :
    def get_priorprobability():
      P_y_eq_k = []
      for x in self.class_ :
        for y in x :
          p_y_eq_k = [np.sum(self.training_data_Y == y) / len(self.training_data_Y)]
          P_y_eq_k.append(p_y_eq_k)
      return P_y_eq_k

    def get_classspecificmeanvector():
      count = 0
      for x in self.class_ :
          id = []
          c_ = 0
          for z in self.training_data_Y :
            if z == x[0] :
              i = 1
              c_ += 1
            else :
              i = 0
            id.append(i)
          if count == 0 :
            s = 1/c_
            classspecificmeanvector = np.matmul( np.matrix(id).dot(s) , self.training_data_X)
            Tp = (np.matrix(id).T * np.matmul( np.matrix(id).dot(s) , self.training_data_X) )
            count += 1
          else :
            classspecificmeanvector = np.insert(classspecificmeanvector,1 ,  np.matmul(np.matrix(id).dot(1/c_) , self.training_data_X), axis=0)
            Tp += np.matrix(id).T * np.matrix(id).dot(s) * self.training_data_X
            x_to_mean = (Tp - self.training_data_X)
            count += 1
      return classspecificmeanvector , x_to_mean

    def get_sigma():
      sigma = (1/(self.x_to_mean.shape[0] - self.class_.shape[0] ) )* self.x_to_mean.T * self.x_to_mean
      return sigma


    # Linear regression module init
    self.training_data_X = training_data_X # The training data x => features numpy_matrix
    self.training_data_Y = training_data_Y # The training data y => response numpy_matrix
    self.class_ = np.unique(self.training_data_Y, axis=0)
    self.prioprobability = get_priorprobability()
    self.classspecificmeanvector, self.x_to_mean = get_classspecificmeanvector()
    self.sigma = get_sigma()

  def predict(self , x):
      #print(self.training_data_X.T , x.T)
      s =  ( self.classspecificmeanvector * inv(self.sigma) * x.T ) + 0.5 * ((self.classspecificmeanvector * inv(self.sigma) ) * self.classspecificmeanvector.T) + np.log(self.prioprobability)
      return self.class_[np.argmax(np.sum(s,axis=1))][0]

if __name__ == "__main__" :
  x = np.matrix([[1,3],[2,3],[2,4],[3,1],[3,2],[4,2] ])
  y = np.matrix([[1],[1],[1],[2],[2],[2] ] )
  Lda = lineardiscriminantananlysis(x,y)
  #print(Lda.class_)
  #print(Lda.prioprobability)
  #print(Lda.classspecificmeanvector)
  #print(Lda.sigma)
  x_ = np.matrix([ [0,5] ]  )
  Lda.predict(x_)
  x_ = np.matrix([ [3,0] ]  )
  Lda.predict(x_)

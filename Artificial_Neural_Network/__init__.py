import numpy as np
from Artificial_Neural_Network.linearregression import linearregression as LR
from Artificial_Neural_Network.logisticregession import logisticregression as Lgr
from Artificial_Neural_Network.lineardiscriminantanalysis import lineardiscriminantananlysis as Lda
from Artificial_Neural_Network.tanhiperboliqueregression import tanhiperboliqueregression as Thr
from Artificial_Neural_Network.artificialneuralnetwork.softmaxregression import softmaxregression as sftmax

class artificialneuralnetwork_classifier :


  def __init__ (self,training_data_X, training_data_Y) :

    def apply_classification ():
      k = 0
      for i in self.training_data_X :
        for j in i :
          if k == 0 :
            zi = np.matrix([[ THR.predict(j), LGR.predict(j) ]])
          else :
            zi = np.insert(zi , 0 , np.matrix([[THR.predict(j), LGR.predict(j)  ]]) , axis=0)
          k += 1
      zi = np.flip(zi,0)
      LDA = Lda(zi,self.training_data_Y)
      return LDA



    pading = np.ones(training_data_X.shape[0])
    #self.training_data_X = np.insert(training_data_X, 0, pading, axis=1) # The training data x => features numpy_matrix
    self.training_data_X = training_data_X
    self.training_data_Y = training_data_Y # The training data y => response numpy_matrix
    Lr = LR(self.training_data_X,self.training_data_Y)
    alpha_1, rss = Lr.leastsquare()
    try :
      LGR = Lgr(self.training_data_X,self.training_data_Y)
      alpha_2 = LGR.Beta
    except :
      alpha_2 = np.zeros((alpha_1.shape[0],alpha_1.shape[1] ) )
      pass
    try :
      THR = Thr(self.training_data_X,self.training_data_Y)
      alpha_3 = THR.Beta
    except :
      alpha_3 = np.zeros((alpha_1.shape[0],alpha_1.shape[1] ) )
      pass
    try :
      SFTMAX = sftmax(self.training_data_X,self.training_data_Y)
      alpha_4 = SFTMAX.Beta
    except :
      alpha_4 = np.zeros((alpha_1.shape[0],alpha_1.shape[1] ) )
      pass
    LDA = apply_classification ()
    self.THR = THR
    self.LGR = LGR
    self.LDA = LDA

  def predict(self,x):
    zi = np.matrix([[ self.THR.predict(x), self.LGR.predict(x) ]])
    return self.LDA.predict(zi)




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

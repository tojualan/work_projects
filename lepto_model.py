import numpy as np
import pandas as pd
import global_functions as gl

class model0:
    """
    Model for two RHNs + singlet scalar
    """

    def __init__(self, M1 = 2000, M2=4000, K1 = 7.8, K2 = 46, K12 = 390000,
                 beta = 100, alpha1 = 0.0015, alpha2 = 0.0015, MS = 1000):
        """
        Initialize model parameters
        """

        self.M1 = M1
        self.M2 = M2
        self.MS = MS
        self.K1 = K1
        self.K2 = K2
        self.K12 = K12
        self.beta = beta
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def mi(self,i):
        if i==1:
            return self.M1
        elif i==2:
            return self.M2
        else:
            return 0

    def ki(self,i):
        if i==1:
            return self.K1
        elif i==2:
            return self.K2
        else:
            return 0

    def rji(self,i,j):
        return self.mi(j)/self.mi(i)

    def sigi(self,i):
        return self.MS**2/self.mi(i)**2

    def gij(self,i,j):
        return ((np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i))
                        +4*self.rji(i,j)*self.sigi(i)+2*self.sigi(i))
                +np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i)))
                *np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i))
                         +4*self.rji(i,j).self.sigi(i))
                -np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i))))/
                (np.sqrt(gl.delta(1, self.rji(i, j), self.sigi(i))
                         + 4 * self.rji(i, j) * self.sigi(i) + 2 *self.sigi(i))
                 - np.sqrt(gl.delta(1, self.rji(i, j), self.sigi(i)))
                 * np.sqrt(gl.delta(1, self.rji(i, j), self.sigi(i))
                           + 4 * self.rji(i, j).self.sigi(i))
                 + np.sqrt(gl.delta(1, self.rji(i, j), self.sigi(i))))
                )

    def get_alpha12(self):
        """
        off-diagonal Yukawa
        off-diagonal Yukawa
        """
        al12 = np.sqrt(self.K12*(self.M2/(16*np.pi)*((1+self.M1/self.M2)**2-
                                                     self.MS**2/self.M2**2)
                                 *np.sqrt((1-self.M1**2/self.M2**2
                                           -self.MS**2/self.M1**2)**2
                                          -4*(self.M1/self.M2
                                              *self.MS/self.M2)**2)
                                 /gl.hub(self.M2))**(-1))
        self.alpha12 = al12

        return 0

    def get_epsilons(self):
        pass

if __name__=='__main__':
    m0 = model0()
    m0.get_alpha12()
    print(gl.geff(100))
    print(m0.alpha12)



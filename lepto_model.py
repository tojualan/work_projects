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

        #calculate the off-diagonal Yukawa + epsilons
        self.get_alpha12()
        self.get_epsilons()

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
        return self.mi(j)**2/self.mi(i)**2

    def sigi(self,i):
        return self.MS**2/self.mi(i)**2

    def gij(self,i,j):
        return ((np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i))
                        +4*self.rji(i,j)*self.sigi(i)+2*self.sigi(i))
                +np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i)))
                *np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i))
                         +4*self.rji(i,j)*self.sigi(i))
                -np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i))))/
                (np.sqrt(gl.delta(1, self.rji(i, j), self.sigi(i))
                         + 4 * self.rji(i, j) * self.sigi(i) + 2 *self.sigi(i))
                 - np.sqrt(gl.delta(1, self.rji(i, j), self.sigi(i)))
                 * np.sqrt(gl.delta(1, self.rji(i, j), self.sigi(i))
                           + 4 * self.rji(i, j)*self.sigi(i))
                 + np.sqrt(gl.delta(1, self.rji(i, j), self.sigi(i))))
                )
    def fijLL(self, i, j):
        return (-np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i)))
                +self.rji(i,j)*np.log(self.gij(i,j)))

    def fijRL(self, i, j):
        return (-np.sqrt(self.rji(i,j))*np.sqrt(gl.delta(1,self.rji(i,j),
                                                         self.sigi(i)))
                +np.sqrt(self.rji(i,j))*np.log(self.gij(i,j)))
    def fijlLL(self,i,j,l):
        return (0.5*np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i)))
                *(np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i))
                          +4*self.rji(i,j))/(1-self.rji(i,l))))
    def fijlLR(self,i,j,l):
        return (np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i)))
                *(np.sqrt(self.rji(i,j)*self.rji(i,l))/(1-self.rji(i,l))))

    def fijlRL(self,i,j,l):
        return (np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i)))
                *(np.sqrt(self.rji(i,j))/(1-self.rji(i,l))))

    def fijlRR(self,i,j,l):
        return (0.5*np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i)))
                /(1-self.rji(i,l))*np.sqrt(self.rji(i,l))
                *np.sqrt(gl.delta(1,self.rji(i,j),self.sigi(i))
                         +4*self.rji(i,j)))

    def f0v(self,i,j):
        return (np.sqrt(self.rji(i,j))
                *(1-(1+self.rji(i,j))*np.log((1+self.rji(i,j))/self.rji(i,j))))

    def f0w(self,i,j):
        return np.sqrt(self.rji(i,j))/(1-self.rji(i,j))

    def get_alpha12(self):
        """
        off-diagonal Yukawa
        off-diagonal Yukawa
        """
        al12 = np.sqrt(self.K12*(self.M2/(16*np.pi)*((1+self.M1/self.M2)**2
                                                     -self.MS**2/self.M2**2)
                                 *np.sqrt((1-self.M1**2/self.M2**2
                                           -self.MS**2/self.M2**2)**2
                                          -4*(self.M1/self.M2
                                              *self.MS/self.M2)**2)
                                 /gl.hub(self.M2))**(-1))
        self.alpha12 = al12

        return 0

    def get_epsilons(self):
        """
        asymmetry coefficients, v-vertex, w-wave function
        """
        self.eps1 = 3*self.M1*gl.Smnu/(8*np.pi*gl.vw**2)
        self.eps20 = (self.M1*gl.Smnu/(8*np.pi*gl.vw**2)\
                      *np.abs(self.f0v(2,1)+self.f0w(2,1)))
        self.eps2v = (self.alpha12*self.beta/(8*np.pi*self.M2)
                      *np.sqrt(self.M1/self.M2)*(self.fijLL(2,1)
                                                 +self.fijRL(2,1)))
        self.eps2w = ((self.alpha1*self.alpha12)/(8*np.pi)
                      *np.sqrt(self.M1/self.M2)
                      *(self.fijlLL(2,1,1)+self.fijlRL(2,1,1)
                        +self.fijlLR(2,1,1)+self.fijlRR(2,1,1)))
        self.eps2vw0 = self.eps20+np.abs(self.eps2v+self.eps2w)

        return 0



if __name__=='__main__':
    m0 = model0()
    # m0.get_epsilons()
    print(m0.alpha12)
    print('\n')

    print(m0.eps20)
    print(m0.eps2v)
    print(m0.eps2w)




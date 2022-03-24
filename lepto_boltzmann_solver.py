import lepto_model as lm
import scipy.special
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

class boltzmann:
    """
    Boltzmann solver
    """
    def __init__(self,model,**kwargs):
        self.m = model
        self.YLeq = 3 / 4

    def Yeq(self,z):
        return 3/8*z**2*scipy.special.kn(2,z)

    def D(self,z,K):
        return K*z**2*self.Yeq(z)*scipy.special.kn(1,z)/scipy.special.kn(2,z)

    def W(self,z,K):
        return 0.5*self.D(z,K)/self.YLeq

    def SW(self,z):
        return 0

    def S1(self,z):
        return 0

    def S2(self,z):
        return 0

    def BY1(self,z,y):
        return (-(y[0]/self.Yeq(z)-1)*(self.D(z,self.m.K1)
                                    +self.D(self.m.M2/self.m.M1*z,self.m.K12)
                                    +self.S1(z))
                +(y[1]/self.Yeq(self.m.M2/self.m.M1*z)-1)
                *(self.D(self.m.M2/self.m.M1*z,self.m.K12))
                -(y[0]*y[1]/(self.Yeq(z)*self.Yeq(self.m.M2/self.m.M1*z))-1)
                *self.SW(z)
                -(y[0]**2/self.Yeq(z)**2-1)*self.SW(z)
                )
    def BY2(self,z,y):
        return (-(y[1]/self.Yeq(self.m.M2/self.m.M1*z)-1)
                *(self.D(self.m.M2/self.m.M1*z,self.m.K2)
                  +self.D(self.m.M2/self.m.M1*z,self.m.K12)+self.S2(z))
                +(y[0]/self.Yeq(z)-1)*self.D(self.m.M2/self.m.M1*z,self.m.K12)
                -(y[0]*y[1]/(self.Yeq(z)*self.Yeq(self.m.M2/self.m.M1*z))-1)
                *self.SW(z)
                -(y[1]**2/self.Yeq(self.m.M2/self.m.M1*z)**2-1)*self.SW(z)
                )

    def BYL(self,z,y):
        return (self.m.eps1*self.D(z,self.m.K1)*(y[0]/self.Yeq(z)-1)
                +self.m.eps2vw0*self.D(self.m.M2/self.m.M1*z,self.m.K2)
                *(y[1]/self.Yeq(self.m.M2/self.m.M1*z)-1)
                -y[2]*(self.W(z,self.m.K1)
                     +self.W(self.m.M2/self.m.M1*z,self.m.K2)))

    def ode_RHS(self,z,y):
        return [self.BY1(z,y)/z,self.BY2(z,y)/z,self.BYL(z,y)/z]

    def solve_boltz(self, zrange = (0.001,20), ivs=None):
        if ivs is None:
            ivs = [self.Yeq(0.001), self.Yeq(0.001), 0]
            print(ivs)
        return scipy.integrate.solve_ivp(self.ode_RHS, zrange, ivs,
                                         method = 'Radau')

    # def ode_RHS2(self,z,y):
    #     return [y[0]/y[1]**2,z**2,y[0]**0.5]
    #
    # def solve_boltz0(self, zrange = (1,10), ivs=None):
    #     if ivs is None:
    #         ivs = [3, .2, 0.1]
    #     return scipy.integrate.solve_ivp(self.ode_RHS2, zrange, ivs)


    def describe(self):
        print(f'Model parameters:\nM1 = {self.m.M1}, M2 = {self.m.M2}, '
              f'MS = {self.m.MS},\nK1 = {self.m.K1}, K2 = {self.m.K2},'
              f' K12 = {self.m.K12},\nalph1 = {self.m.alpha1}, '
              f' alph2 = {self.m.alpha2},\nalph12 = {self.m.alpha12:.5},'
              f' beta = {self.m.beta},\n')

        print(f'Epsilons:\neps1_0 = {self.m.eps1:.5}')
        print(f'eps2_0 = {self.m.eps2v:.5}')
        print(f'eps2_v = {self.m.eps2v:.5}')
        print(f'eps2_w = {self.m.eps2w:.5}')
        print(f'eps2 = {self.m.eps2vw0:.5}')

if __name__=='__main__':

    boltz = boltzmann(lm.model0())
    boltz.describe()
    print(boltz.ode_RHS(0.001,[3/4.,3/4.,0]))
    res = boltz.solve_boltz()
    print(res.t)
    print(res.y[0]-boltz.Yeq(res.t))
    plt.figure()
    plt.loglog(res.t,res.y[0],label='y1')
    plt.loglog(res.t, res.y[1],label='y2')
    plt.loglog(res.t, 1e8*res.y[2],label='yL')
    plt.loglog(res.t,0.11*np.ones(len(res.t)),'--')
    plt.legend()
    plt.ylim(1e-3,100)
    plt.show()



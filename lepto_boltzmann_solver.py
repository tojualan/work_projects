import lepto_model as lm

class boltzmann:
    """
    Boltzmann solver
    """
    def __init__(self,model,**kwargs):
        self.m = model


    def ode_RHS(self):
        pass

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

if __name__=='__main__':

    boltz = boltzmann(lm.model0())
    boltz.describe()
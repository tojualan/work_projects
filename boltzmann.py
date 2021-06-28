
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
from scipy import integrate
from scipy import interpolate

MPl = 1.22e19
#g = 100.
#gs = 100.
mh = 125.
Gammah = 4.07e-3
mW = 80.4
mZ = 91.2
mb = 4.2
vw = 246.22

#effective dof's
df = pd.read_csv('eff_dof.dat',delimiter=' ', comment='#')
df['temp']=df['Temperature(MeV)']/1000
geff = interpolate.interp1d(df.temp,df.geff)
geff_s = interpolate.interp1d(df.temp,df.geff_s)


class ScalarDMModel:
    """
    Creates the scalar DM model and the relevant cross sections
    """
    def __init__(self,**kwargs):
        if 'mDM' in kwargs: self.mDM = kwargs['mDM']
        else: print('No DM mass given')
        if 'ghhh' in kwargs: self.ghhh = kwargs['ghhh']
        else: self.ghhh = 3*mh**2/vw


    def sigma_hh(self, s,lam):
        """
        :param s: Mandelstam s
        :param lam: scalar portal coupling
        :return:  cross section ss -> hh
        """
        if self.mDM<mh:
            return 0
        return (np.sqrt(s-4*mh**2)*np.sqrt(s-4*self.mDM**2)/(32*np.pi*s)
                *(np.abs(lam + lam*vw*self.ghhh
                         /(s-mh**2+1j*mh*Gammah))**2
                  +16*(lam*vw)**4/(s-2*mh**2)**2
                  -4*(lam*vw)**2*(2*lam-2*self.ghhh*lam*vw*(mh**2-s)
                                   /(s**2+mh**4+mh**2*(Gammah**2-2*s)))
                  /(s-2*mh**2)))

    def sigma_WW(self,s,lam):
        """
        :param s: Mandelstam s
        :param lam: scalar portal coupling
        :return:  cross section ss -> WW
        """
        if self.mDM<mW:
            return 0
        return (np.sqrt(s-4*self.mDM**2)*np.sqrt(s-4*mW**2)*(lam*vw)**2
                *(12*mW**4-4*mW**2*s+s**2)
                /(16*np.pi*s*vw**2*(mh**4+mh**2*(Gammah**2-2*s)+s**2)))

    def sigma_ZZ(self,s,lam):
        """
        :param s: Mandelstam s
        :param lam: scalar portal coupling
        :return:  cross section ss -> ZZ
        """
        if self.mDM<mZ:
            return 0
        return (np.sqrt(s-4*self.mDM**2)*np.sqrt(s-4*mZ**2)*(lam*vw)**2
                *(12*mZ**4-4*mZ**2*s+s**2)
                /(32*np.pi*s*vw**2*(mh**4+mh**2*(Gammah**2-2*s)+s**2)))

    def sigma_bb(self,s,lam):
        """
        :param s: Mandelstam s
        :param lam: scalar portal coupling
        :return:  cross section ss -> bb
        """
        if self.mDM<mb:
            return 0
        return (3*(lam*vw)**2*mb**2*np.sqrt(s-4*mb**2)**3
                *np.sqrt(s-4*self.mDM**2)
                /(8*np.pi*s*vw**2*(mh**4+mh**2*(Gammah**2-2*s)+s**2)))

    def sigma(self,s,lam):
        return (self.sigma_hh(s,lam) + self.sigma_WW(s,lam)
                + self.sigma_ZZ(s,lam) + self.sigma_bb(s,lam))

    def sigmaTh(self, s, x, lam):
        """
        :param s: Mandelstam s
        :param x: mDM/T
        :param lam: scalar portal coupling
        :return:  Integrand for the thermally averaged cross section
        """
        if s<4*self.mDM**2:
            return 0
        return (1/(8*self.mDM**4*self.mDM/x*special.kn(2,x)**2)*np.sqrt(s)
                *self.sigma(s,lam)*special.kn(1,np.sqrt(s)*x/self.mDM))

    def sigmavT(self, x, lam):
        """
        Integration over s for the thermally averaged cross section
        :param x: mDM/T
        :param lam: scalar portal coupling
        :return: thermally averaged cross section
        """
        smin = 4*self.mDM**2
        smax = 100*self.mDM**2 #np.inf
        return integrate.quad(self.sigmaTh, smin, smax,
                              args=(x,lam), epsabs=1e-25, epsrel=1e-5)[0]

    def interpolate_thermal_xsec(self, lam):
        """
        Generates the interpolation of the thermally averaged cross section
        :param lam: scalar portal coupling
        :return: interpolation
        """
        # print('Generate the interpolation of sigmav')
        x_span = np.arange(2., 105, 0.5)
        y_vals = [self.sigmavT(x,lam) for x in x_span]
        return interpolate.interp1d(x_span, y_vals)


class Boltzmann(ScalarDMModel):
    """
    Creates the DM model, and calculates the relevant cross sections
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if 'gsshh' in kwargs: self.gsshh = kwargs['gsshh']
        else: self.gsshh = 0
        if 'x_range' in kwargs: self.x_range = kwargs['x_range']
        else: self.x_range = [3,100]
        if 'y0' in kwargs: self.y0 = kwargs['y0']
        else: self.y0 = [self.Yeq(3)]
        if 'x_eval' in kwargs: self.x_eval = kwargs['x_eval']
        else: self.x_eval = np.arange(*self.x_range,0.1)

        self._svTintpd = super().interpolate_thermal_xsec(self.gsshh)
        self.find_xf()
        self.get_abundance()
        self.sol = self.solve_ivp()


    def Yeq(self, x):
        """
        :param x: mDM/T
        :return: Equilibrium yield
        """
        # return (45/(2*np.pi**4)*np.sqrt(np.pi/8.)
        #         *geff(self.mDM/x)/geff_s(self.mDM/x)*x**(3/2)*np.exp(-x))
        return (x**2*special.kn(2,x)*45/(4*np.pi**4*geff_s(self.mDM/x)))

    def Z(self, x):
        """
        :param x: mDM/T
        :return: front factor in the ode
        """

        return (-(2*np.pi**2/45*MPl/1.66
                *geff_s(self.mDM/x)/np.sqrt(geff(self.mDM/x))*self.mDM)
                *self._svTintpd(x)/x**2)

    def ode(self, x, y):
        """
        :param x: mDM/T
        :param y: DM yield
        :return: the rhs of the ode dy/dx = ode(x,y)
        """
        return self.Z(x)*(y**2 - self.Yeq(x)**2)

    def solve_ivp(self):
        """
        numerical solution of the ode
        """
        return integrate.solve_ivp(self.ode, self.x_range, self.y0,
                                   t_eval=self.x_eval, method='Radau',
                                   rtol=1e-6, atol=1e-15)

    def find_xf(self):
        """
        Find the freeze-out temperature, and store it in self.xf
        """
        sol = self.solve_ivp()

        # find the index at which the departure from equilibrium happens
        # from left-hand side
        err = 0
        i = 0
        while err<0.5 and i<len(sol.t)-1:
            err = (sol.y[0][i]-self.Yeq(sol.t[i]))/self.Yeq(sol.t[i])
            #print(_err)
            i +=1
        iL = i

        # find the index at which the departure from equilibrium happens
        # from right-hand side
        err = 1
        i = len(sol.t)-1
        while err > 0.5 and i > 0:
            err = (sol.y[0][i] - self.Yeq(sol.t[i])) / self.Yeq(sol.t[i])
            # print(_err)
            i -= 1
        iR = i

        # take the largest of these (if some numerical instabilities in either
        # direction)
        self.xf = sol.t[max(iR,iL)]

    def get_abundance(self):
        """
        store the DM abundance in self.Omega
        """
        if not self.xf:
            self.find_xf()

        self.Omega = 8.6e-11*self.xf/(np.sqrt(geff_s(self.mDM/self.xf))
                                      *self._svTintpd(self.xf))

class CouplingSolver(ScalarDMModel):
    """
    Solves the coupling producing the given amount of observed DM abundance,
    default frel = 1
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'x_range' in kwargs: self.x_range = kwargs['x_range']
        else: self.x_range = [3,100]
        if 'y0' in kwargs: self.y0 = kwargs['y0']
        else: self.y0 = [self.Yeq(3)]
        if 'x_eval' in kwargs: self.x_eval = kwargs['x_eval']
        else: self.x_eval = np.arange(*self.x_range,0.1)
        if 'frel' in kwargs: self.frel = kwargs['frel']
        else: self.frel = 1
        if 'verbose' in kwargs: self.verbose = kwargs['verbose']
        else: self.verbose = False

    def Yeq(self, x):
        """
        Equilibrium yield as a function of x
        """
        # return (45/(2*np.pi**4)*np.sqrt(np.pi/8.)
        #         *geff(self.mDM/x)/geff_s(self.mDM/x)*x**(3/2)*np.exp(-x))
        return (x**2*special.kn(2,x)*45/(4*np.pi**4*geff_s(self.mDM/x)))

    def Z(self, x):
        if not self._svTintpd:
            print('Interpolation missing')
            return 0
        return (-(2*np.pi**2/45*MPl/1.66
                *geff_s(self.mDM/x)/np.sqrt(geff(self.mDM/x))*self.mDM)
                *self._svTintpd(x)/x**2)

    def ode(self, x, y):
        return self.Z(x)*(y**2 - self.Yeq(x)**2)

    def solve_ivp(self):
        return integrate.solve_ivp(self.ode, self.x_range, self.y0,
                                   t_eval=self.x_eval, method='Radau',
                                   rtol=1e-6, atol=1e-15)

    def find_xf(self):
        """
        Find the freeze-out temperature
        :param x_range:
        :param y0:
        :return:
        """
        sol = self.solve_ivp()

        # find the index at which the departure from equilibrium happens
        # from left-hand side
        err = 0
        i = 0
        while err<0.5 and i<len(sol.t)-1:
            err = (sol.y[0][i]-self.Yeq(sol.t[i]))/self.Yeq(sol.t[i])
            i +=1
        iL = i

        # find the index at which the departure from equilibrium happens
        # from right-hand side
        err = 1
        i = len(sol.t)-1

        while err > 0.5 and i > 0:
            err = (sol.y[0][i] - self.Yeq(sol.t[i])) / self.Yeq(sol.t[i])
            # print(_err)
            i -= 1
        iR = i

        # take the largest of these (if some numerical instabilities in either
        # direction)
        self.xf = sol.t[max(iR,iL)]

    def get_abundance(self):
        """
        DM abundance
        """
        if not self.xf:
            self.find_xf()
        if not self._svTintpd:
            print('Interpolation missing')
            return 0
        else:
            self.Omega = 8.6e-11*self.xf/(np.sqrt(geff_s(self.mDM/self.xf))
                                          *self._svTintpd(self.xf))
            return self.Omega

    def find_coupling(self):
        """
        finds the coupling that produces the required abuncance
        """
        err0 = 1
        llam0 = -2
        llam1 = -4
        niter = 0
        while err0 > 1e-3 and niter<50:
            self._svTintpd = super().interpolate_thermal_xsec(np.exp(llam0))
            self.find_xf()
            omega0 = self.get_abundance()

            if omega0 > self.frel * 0.12:
                llam0 = 0.5*llam0
                continue

            self._svTintpd = super().interpolate_thermal_xsec(np.exp(llam1))
            self.find_xf()
            omega1 = self.get_abundance()

            if omega1 < self.frel * 0.12:
                llam1 = 2*llam1
                continue

            llam2 = 0.5*(llam0 + llam1)
            self._svTintpd = super().interpolate_thermal_xsec(np.exp(llam2))
            self.find_xf()
            omega2 = self.get_abundance()

            if self.verbose:
                print(f'n: {niter}, Omega0: {omega0:.3f}, '
                      f'Omega1: {omega1:.3f}, Omega2: {omega2:.3f}')

            if omega2 > self.frel * 0.12:
                llam1 = llam2
            else:
                llam0 = llam2

            err0 = np.abs(omega2-self.frel*0.12)

            if self.verbose:
                print(f'lam0: {llam0}, lam1: {llam1}')
            niter += 1

        self.lam = np.exp(llam2)
        print(f'The coupling corresponding to frel = {self.frel:.2f} is '
              f'{self.lam:.3e}')






def main():

    model1 = Boltzmann(mDM=300., gsshh=0.1)
    model2 = Boltzmann(mDM=300., gsshh=0.25)
    model3 = Boltzmann(mDM=300., gsshh=0.5)
    #
    print(f'xf = {model1.xf:.2f}, Omega h^2 = {model1.Omega:.2e}, frel = '
          f'{model1.Omega/0.12:.2e}')

    print(f'xf = {model2.xf:.2f}, Omega h^2 = {model2.Omega:.2e}, frel = '
          f'{model2.Omega/0.12:.2e}')

    print(f'xf = {model3.xf:.2f}, Omega h^2 = {model3.Omega:.2e}, frel = '
          f'{model3.Omega/0.12:.2e}')

    #
    # Plot
    #
    fig,ax = plt.subplots()
    ax.plot(model1.sol.t, list(map(model1.Yeq, model1.sol.t)), '-.',
            label=r'$Y_{{\rm eq}}$')
    ax.plot(model1.sol.t, model1.sol.y[0], 'b',
            label = r'$f_{{\rm rel}} = {:.2f}$, '
                    r'$ g_{{sshh}} = {:.4f}$'.format(model1.Omega/0.12,
                                                     model1.gsshh))
    ax.plot(model2.sol.t, model2.sol.y[0], 'purple',
            label= r'$f_{{\rm rel}} = {:.2f}$, '
                   r'$g_{{sshh}} = {:.4f}$'.format(model2.Omega/0.12,
                                                 model2.gsshh))
    ax.plot(model3.sol.t, model3.sol.y[0], 'r',
            label = r'$f_{{\rm rel}} = {:.2f}$, '
                    r'$g_{{sshh}} = {:.4f}$'.format(model3.Omega/0.12,
                                                  model3.gsshh))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.vlines(model1.xf,1e-16,1,'gray','--')
    ax.set_ylim(1e-16, 1)
    ax.set_xlabel('$x = m/T$')
    ax.set_ylabel('$Y$')
    plt.draw()
    ax.set_xticks(list(ax.get_xticks())+[model1.xf])
    ax.set_xticklabels([r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$',
                        r'$10^3$',r'$10^4$',
                        f'$x_f = {model1.xf:.1f}$'])
    ax.set_xlim(3,100)
    ax.grid(alpha=0.5)
    ax.set_title('Higgs mediated scalar DM, '
                 r'$m_{{\rm DM}} = $ {:.0f} GeV'.format(model1.mDM))
    ax.legend(loc='upper right')
    plt.show()



if __name__ == '__main__':
    main()

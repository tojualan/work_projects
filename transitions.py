"""
This part computes the transitions. Data handling parallelised.

Included: d(S/T)/dT. This part requires modified cosmoTransitions source code!!

Just modify the data file names below, number of parallel kernels
and optionally the maximum time spent/point; default 180 s.

Tommi Alanne
Last changed: 15.6.2020

Last modification: Proper diagonalisation of Z-A system at finite T.
"""


file_in = 'fin.csv'
file_out = 'res_fin.csv'
time_cut = 240
num_workers = 2


#some global parameteres (at ren. scale = v0)

v0 = 246.22
gLs = 0.416
gYs = 0.129
yt = 0.922
MPl = 1.22e19

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder
import io
from contextlib import redirect_stdout
import warnings
from multiprocessing import Process
from scipy.interpolate import UnivariateSpline
from multiprocessing import Process, Manager
import itertools

text_trap = io.StringIO()
warnings.filterwarnings("ignore")


#effective dof's
df = pd.read_csv('eff_dof.dat',delimiter=' ', comment='#')
df['temp']=df['Temperature(MeV)']/1000
geff = interp1d(df.temp,df.geff)


class model1(generic_potential.generic_potential):
    """
    Full finite-T 1-loop potential with thermal mass enhancement instead of
    full daisy resummation
    """
    def init(self, m1 = 125.1, m2 = 500., mDM = 200., theta = 0.1,
             w0 = 500., mu1 = 100., muS3 = 100., muS3p = 100.,
             muHS3 = 100., laHSp = 0.0, laSp = 0.0,
             dlaH=0.,dlaS=0.,dlaHS=0.,dmuH2=0.,dmuS2=0.,dmuS2p=0.):

        # The init method is called by the generic_potential class, after it
        # already does some of its own initialization in the default __init__()
        # method. This is necessary for all subclasses to implement.

        # This first line is absolutely essential in all subclasses.
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 2

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = v0**2

        """
        Initialize parameters:
          m1 - higgs mass.
          m2 - heavy scalar mass.
          mDM - DM mass
          w0 - singlet vev at T=0.
          theta - scalar mixing angle
          mu1 - S linear coupling
          muS3 - S^3 trilinear coupling
          muS3 - S^3 CP-odd coupling
          muHS3 - trilinear SH^2 coupling
          lHSp - CP-odd HS portal
          lSp - CP-odd singlet quartic
          dlaH,dlaS, dlaHS,dmuH2,dmuS2,dmuS2p - counter terms

        """
        self.m1 = m1
        self.m2 = m2
        self.mDM = mDM
        self.theta = theta
        self.w0 = w0
        self.mu1 = mu1
        self.muS3 = muS3
        self.muS3p = muS3p
        self.muHS3 = muHS3
        self.laHSp = laHSp
        self.laSp = laSp
        self.dlaH = dlaH
        self.dlaS = dlaS
        self.dlaHS = dlaHS
        self.dmuH2 = dmuH2
        self.dmuS2 = dmuS2
        self.dmuS2p = dmuS2p

        """
        Couplings fixed by tree-level minimisation & diagonalisation:

        laH : H^4 quartic
        laS : S^4 quartic
        laHS : H^2S^2 portal
        muH2 : H mass term
        muS2 : S mass term
        muS2p : eta mass term

        """

        self.laH = (1./(2.*v0**2)*(m1**2 * np.cos(theta)**2
                                 + m2**2 * np.sin(theta)**2))
        self.laS = (1./(8.*w0**3)*(-8.*laSp*w0**3
                                 + 4.*w0*np.sin(theta)**2*(m1**2- m2**2)
                                 + 4.*m2**2*w0+4.*mu1
                                 -3.*np.sqrt(2.)*w0**2*(muS3 + muS3p)
                                 + muHS3*v0**2))
        self.laHS = (-1./(2.*v0*w0)*(2.*laHSp*v0*w0
                                   + np.sin(2.*theta)*(m1**2 - m2**2)
                                   + muHS3*v0))
        self.muH2 = (1./(4.*v0)*(-2.*m1**2*v0*np.cos(theta)**2
                               - 2.*m2**2*v0*np.sin(theta)**2
                               + (m1**2 - m2**2)*w0*np.sin(2.*theta)
                               - muHS3*v0*w0))
        self.muS2 = (0.5*laHSp*v0**2 + 2.*laSp*w0**2
                     + 0.25*(m1**2 - m2**2)*v0*np.sin(2.*theta)/w0
                     - 0.5*(m1**2 - m2**2)*np.sin(theta)**2 - 0.5*m2**2
                     + 0.5*mDM**2 - mu1/w0 + w0*np.sqrt(2.)*(3.*muS3-muS3p)/4.)
        self.muS2p = (-0.5*laHSp*v0**2-2.*laSp*w0**2-0.5*mDM**2
                      - mu1/(2.*w0)-9/8.*w0*np.sqrt(2.)*muS3
                      - muS3p*w0*np.sqrt(2.)/8. - muHS3*v0**2/(8.*w0))



#    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry
        in the theory and you don't want to double-count all of the phases. In
        this case, we're throwing away all phases whose zeroth (since python
        starts arrays at 0) field component of the vev goes below -5. Note that
        we don't want to set this to just going below zero, since we are
        interested in phases with vevs exactly at 0, and floating point numbers
        will never be accurate enough to ensure that these aren't slightly
        negative.
        """
#        return (np.array([X])[...,0] < -5.0).any()


    def V0(self, X):
        """
        This method defines the tree-level potential.
        """
        # X is the input field array. It is helpful to ensure that it is a
        # numpy array before splitting it into its components.
        X = np.asanyarray(X)
        # x and y are the two fields that make up the input. The array should
        # always be defined such that the very last axis contains the different
        # fields, hence the ellipses.
        # (For example, X can be an array of N two dimensional points and have
        # shape (N,2), but it should NOT be a series of two arrays of length N
        # and have shape (2,N).)
        h,s = X[...,0], X[...,1]

        H = 1./np.sqrt(2.) * h
        S = 1./np.sqrt(2.) * s
        Sh = 1./np.sqrt(2.) * s

        r = (self.muH2 * H**2 + self.muS2 * S*Sh + self.laH * H**4
             + self.laS * (S*Sh)**2 + self.laHS * S*Sh * H**2 )
        r += (self.mu1/np.sqrt(2.) * (S + Sh) + self.muS2p/2.*(S**2 + Sh**2)
              + 0.5*self.muS3*(S**3 + Sh**3) + 0.5*self.muS3p*S*Sh*(S + Sh)
              + self.muHS3/ (2.*np.sqrt(2.)) * H**2 * (S + Sh)
              + 0.5*self.laSp * (S**4 + Sh**4)
              + 0.5*self.laHSp * H**2 * (S**2 + Sh**2))

        # include counter terms to keep the vevs and masses fixed at
        #tree-level values!

        r += (self.dmuH2 * H**2 + self.dmuS2 * S*Sh + self.dlaH * H**4
              + self.dlaS * (S*Sh)**2 + self.dlaHS * S*Sh * H**2
              + 0.5*self.dmuS2p * (S**2 + Sh**2))

        return r


    def boson_massSq(self, X, T):
        X = np.array(X)
        h,s = X[...,0], X[...,1]


        #thermal masses
        cS = (2.0 * self.laHS + 4.0 * self.laS + 2.0*self.laHSp) / 12.0
        ceta = (2.0 * self.laHS + 4.0 * self.laS - 2.0*self.laHSp) / 12.0
        cH = ((9.0 * gLs + 3.0 * gYs + 12.0 * yt**2 + 24.0 * self.laH
               + 4.0 * self.laHS) / 48.0)

        #regulating mass for the photon at T=0
        muReg = 1.


        # The field-dependent boson masses.
        # Note that these can also include temperature-dependent corrections.

        a = (3.*self.laH*h**2+0.5*self.laHS*s**2+0.5*self.muHS3*s+self.muH2
             + 0.5*self.laHSp*s**2)
        a += cH*T**2

        b = (0.5*self.laHS*h**2 + 3.*self.laS*s**2
             + 1.5*np.sqrt(2.)*(self.muS3 + self.muS3p)*s
             + self.muS2 + self.muS2p
             + 0.5*self.laHSp*h**2 + 3.*self.laSp*s**2)
        b += cS*T**2

        d = 0.5*(2.*self.laHS*s+self.muHS3+2.*self.laHSp*s)*h

        #h1 & h2
        A = 0.5*(a + b)
        B = np.sqrt(0.25*(a-b)**2 + d**2)

        # eta
        C = (0.5*self.laHS*h**2 + self.laS*s**2
             - 0.5*np.sqrt(2.)*(3.*self.muS3-self.muS3p)*s
             + self.muS2 - self.muS2p
             -0.5*self.laHSp*h**2-3.*self.laSp*s**2)
        C += ceta*T**2


        mWsT = 0.25*gLs*h**2
        mZsT = 0.25*(gLs+gYs)*h**2

        mWsL = mWsT + gLs*T**2*11./6.

        """
        proper diagonalisation of Z-A system in finite T
        """

        a1 = np.sqrt((36.*gLs*gYs*h**4)/((gLs-gYs)**2*(3.*h**2+22.*T**2)**2)+1.)
        b1 = (36.*gYs*h**4)/((gLs-gYs)*(3.*h**2+22.*T**2))

        mZsL = (gLs*(3.*(a1+1)*h**2 + 22.*(a1+1.)*T**2 + b1)
                +gYs*(a1-1.)*(3.*h**2+22.*T**2))/(24.*a1)

        mAsL = muReg + (gLs*(3.*(a1-1.)*h**2 + 22.*(a1-1.)*T**2 - b1)
                        +gYs*(a1+1.)*(3.*h**2 + 22.*T**2))/(24.*a1)


        #EW Goldstones
        mGs = (self.laH*h**2+0.5*self.laHS*s**2+0.5*self.muHS3*s+self.muH2
               +0.5*self.laHSp*s**2)
        mGs += cH*T**2

        M = np.array([A-B, A+B, C, mGs, mWsL, mWsT, mZsL, mZsT, mAsL])


        # At this point, we have an array of boson masses, but each entry might
        # be an array itself. This happens if the input X is an array of points.
        # The generic_potential class requires that the output of this function
        # have the different masses lie along the last axis, just like the
        # different fields lie along the last axis of X, so we need to reorder
        # the axes. The next line does this, and should probably be included in
        # all subclasses.
        M = np.rollaxis(M, 0, len(M.shape))

        # The number of degrees of freedom for the masses. This should be a
        # one-dimensional array with the same number of entries as there are
        # masses.
        dof = np.array([1, 1, 1, 3, 2, 4, 1, 2, 1])

        # c is a constant for each particle used in the Coleman-Weinberg
        # potential using MS-bar renormalization. It equals 1.5 for all scalars
        # and the longitudinal polarizations of the gauge bosons, and 0.5 for
        # transverse gauge bosons.
        c = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 1.5])

        return M, dof, c

    def fermion_massSq(self,X):
        X = np.array(X)
        h,s = X[...,0], X[...,1]

        a = 0.5*yt**2*h**2

        M = np.array([a])

        # At this point, we have an array of boson masses, but each entry might
        # be an array itself. This happens if the input X is an array of points.
        # The generic_potential class requires that the output of this function
        # have the different masses lie along the last axis, just like the
        # different fields lie along the last axis of X, so we need to reorder
        # the axes. The next line does this, and should probably be included in
        # all subclasses.
        M = np.rollaxis(M, 0, len(M.shape))

        # The number of degrees of freedom for the masses. This should be a
        # one-dimensional array with the same number of entries as there are
        # masses.
        dof = np.array([12])

        return M, dof



    def approxZeroTMin(self):
#        # There are generically two minima at zero temperature in this model,
#        # and we want to include both of them.
        return [np.array([-v0,self.w0]), np.array([v0,self.w0])]

    def DVtot(self, X, T):
        """
        The finite temperature effective potential, but offset
        such that V(0, T) = 0.
        """
        #print(X.shape)
        #X0 = np.zeros(self.Ndim)
        X0 = np.zeros(X.shape)
        return self.Vtot(X,T,False) - self.Vtot(X0,T,False)


def Hub(T):
    """
    Hubble parameter as function of T
    """
    return T**2*np.sqrt(8.*np.pi**3*geff(T)/(90.*MPl**2))

def rhoR(T):
    """
    Radiation density
    """
    return geff(T)*np.pi**2*T**4/30.

def derST(STtmp,Tn,phases,start_phase,Vtot,gradV):
    """
    calculates d(S/T)/dT
    """
    # remove entries where S=inf
    STtmp0 = STtmp[np.logical_not(np.isinf(STtmp[:,1]))]
    # recast the list in (T, S/T) format
    STlst = np.array([[x[0],x[1]/x[0]] for x in STtmp0])
    # add middle points in T to refine the fit
    Ttmp1 = [0.5*(STlst[i,0]+STlst[i+1,0]) for i in range(len(STlst)-1)]
    #iterate once more
    Ttmp2 = [0.5*(Ttmp1[i]+Ttmp1[i+1]) for i in range(len(Ttmp1)-1)]
    Ttmp = Ttmp1 + Ttmp2

    tmplst = []

    # get the S/T for the new T values
    for y in Ttmp:
        tmp = transitionFinder._tunnelFromPhaseAtT(
                y,phases,start_phase,Vtot,gradV,phitol=1e-8,
                overlapAngle=45.0,nuclCriterion=lambda S,T: S/(T+1e-100),
                fullTunneling_params={},verbose=False, outdict = {})
        try:
            tmplst.append([y,tmp])
        except:
            pass

    lstAll = np.sort(np.concatenate((STlst,np.array(tmplst)),axis=0),0)

    #interpolate & get the derivative of the interpolation
    spl = UnivariateSpline(lstAll[:,0],lstAll[:,1],k=4)
    der_spl = spl.derivative()

    return der_spl(Tn)


def get_alpha(Tn,Drho,DV):
    """
    Calculates alpha = 1/rho(Delta V - T/4 Delta dV/dT)
                     = 1/(4 rho)(3 Delta V + Delta rho)
    """
    return (3.*DV + Drho)/(4.*rhoR(Tn))


def get_beta(Tn,dSTT):
    """
    Calculates beta = H(Tn)*Tn*d(S/T)/dT
    """

    return Hub(Tn)*Tn*dSTT




def do_work(in_queue):
    """
    Evaluates the phase transitions of a given data point
    from the queue and writes the positive results into a file
    """

    while True:
        item = in_queue.get()
        line_no, line = item

        # exit signal
        if line == None:
            return

        if line_no > 0:
            with redirect_stdout(text_trap):
                #initialise
                vals = line.split(',')[:17]
                m1t,m2t,mDMt,thetat,w0t,mu1t,muS3t,muS3pt,muHS3t,laHSpt,laSpt,\
                        dlaHt,dlaSt,dlaHSt,dmuH2t,dmuS2t,dmuS2pt\
                        = [float(val) for val in vals]

                #print(m1t,m2t,mDMt,thetat,w0t,mu1t,muS3t,muS3pt,muHS3t,laHSpt,
                #      laSpt,dlaHt,dlaSt,dlaHSt,dmuH2t,dmuS2t,dmuS2pt)
                m = model1(m1 = m1t, m2 = m2t, mDM = mDMt, theta = thetat,
                           w0 = w0t, mu1 = mu1t, muS3 = muS3t, muS3p = muS3pt,
                           muHS3 = muHS3t, laHSp = laHSpt, laSp =laSpt,
                           dlaH=dlaHt, dlaS=dlaSt, dlaHS=dlaHSt, dmuH2=dmuH2t,
                           dmuS2=dmuS2t, dmuS2p=dmuS2pt)

                laHt,laSt,laHSt,muH2t,muS2t,muS2pt = (m.laH, m.laS, m.laHS,
                                                      m.muH2, m.muS2, m.muS2p)
                try:
                    #calculate the transitions
                    m.findAllTransitions()
                    mTr = m.TnTrans
                    #print(mTr)
                    mlen = len(mTr)
                    for x in range(mlen):
                        #pick only 1st-order transitions
                        if mTr[x]['trantype'] == 1:
                            #exclude (0,s1) -> (0,s2) transitions and
                            #numerical artefacts
                            #if (np.abs(mTr[x]['low_vev'][0])>5. and
                            #    np.abs(mTr[x]['high_vev'][0])<5. and
                            if ((np.abs(mTr[x]['low_vev'][0]
                                       -mTr[x]['high_vev'][0])
                                +np.abs(mTr[x]['low_vev'][1]
                                       -mTr[x]['high_vev'][1]))>30.):
                                # get d(S/T)/dT
                                STtmp = np.array(m.ncritdict[x])
                                Tnt = mTr[x]['Tnuc']


                                try:
                                    dSTT = derST(
                                            STtmp,Tnt,
                                            m.phases,
                                            m.phases[mTr[x]['high_phase']],
                                            m.Vtot,m.gradV)

                                    alpha = get_alpha(Tnt,mTr[x]['Delta_rho'],
                                                      mTr[x]['Delta_p'])
                                    beta = get_beta(Tnt,dSTT)
                                    betaH = beta/Hub(Tnt)

                                except:
                                    dSTT = np.nan
                                    alpha = np.nan
                                    beta = np.nan


                                #write the results to file
                                with open(file_out,'a') as fout:
                                    fout.write(','.join(line.strip(
                                               ).split(',')[:21])
                                               + ',%.2f,%.2f,%.2f,'
                                               %(laHt, laSt, laHSt)
                                               + '%.2f,%.2f,%.2f,'
                                               %(muH2t,muS2t, muS2pt)
                                               + '%.2f,%.2f,%.1f,%.1f,%.1f,'
                                               %(mTr[x]['crit_trans']['Tcrit'],
                                                 mTr[x]['Tnuc'],
                                                 mTr[x]['high_vev'][0],
                                                 mTr[x]['high_vev'][1],
                                                 mTr[x]['low_vev'][0])
                                               +'%.1f,%.2f,%.2f,%.2f,'
                                               %(mTr[x]['low_vev'][1],
                                                 mTr[x]['action'],
                                                 mTr[x]['Delta_rho'],
                                                 mTr[x]['Delta_p'])
                                               +'%.8f,%.8f,%.14f,%.8f\n'
                                               %(dSTT,alpha,beta,betaH))

                except:
                    pass

            if line_no%10==0:
                print(line_no)





def scanner_gen(file_in, file_out):
    """
    The scanning routine.
    """

    with open(file_out,'w+') as fout:
        fout.write('m1,m2,mDM,theta,w0,mu1,muS3,muS3p,muHS3,laHSp,laSp,dlaH,'
                   + 'dlaS,dlaHS,dmuH2,dmuS2,dmuS2p,xf,Omega_h2,sigma_SI,'
                   + 'sigma_v,laH,laS,laHS,muH2,muS2,muS2p,'
                   + 'Tc,Tn,high_vev_h,high_vev_s,low_vev_h,low_vev_s,'
                   + 'Sn,Delta_rho,Delta_p,dSTT,alpha,beta,beta_over_H\n')

    #setup the parallel data handling
    manager = Manager()
    work = manager.Queue(num_workers)

    # start for workers
    pool = []
    for i in range(num_workers):
        p = Process(target=do_work, args={work})
        p.start()
        pool.append(p)

    # produce data
    with open(file_in) as f:
        iters = itertools.chain(f, (None,)*num_workers)
        for num_and_line in enumerate(iters):
            work.put(num_and_line)

    for p in pool:
        p.join(timeout=time_cut)
        p.terminate()



#run the scanning routine
if __name__ == "__main__":
    scanner_gen(file_in, file_out)

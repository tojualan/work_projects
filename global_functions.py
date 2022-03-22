import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


#effective dof's
df = pd.read_csv('eff_dof.dat',delimiter=' ', comment='#')
df['temp']=df['Temperature(MeV)']/1000
geff = interp1d(df.temp,df.geff)

MPl = 1.22e19
vw = 174
mt = 173.1
Smnu = 0.05e-9
Nc = 3
gstar = 110.75

mstar = 8*np.pi*vw*2*np.sqrt(8*np.pi**3*gstar/(90*MPl**2))

def hub(T):
    return T**2*np.sqrt(8*np.pi**3*geff(T)/(90*MPl**2))


# work_projects
Here are a couple of work examples from previous particle physics research projects. 

---- boltznann.py calculates numerically the dark matter abundance for a real singlet-scalar extension of the Standard Model. It initialises the model with relevant cross sections, then solves the relevant Boltzmann equation (ODE), and finally finds the freeze-out temperature, where the dark matter number density departs from thermal equilibrium. The ODE is stiff, so I use 'Radau' ode solver method. In the end the code prints the results and plots the solutions for the dark matter yield and the corresponding equilibrium curve. Needs the eff_dof.dat file that gives tabulates the number of effective degrees of freedom as a function of temperature.


---- transitions.py finds cosmic phase transitions for complex scalar extension of the Standard Model using the CosmoTransitions python package. Finding the derivative d(S/T)/dT needs modifications in the CosmoTransitions source code, ask for details if interested. The code takes the model parameters as an input csv file and returns the results in csv format. The data handling is parallelised using multiprocessing python package. Needs the eff_dof.dat file that gives tabulates the number of effective degrees of freedom as a function of temperature.

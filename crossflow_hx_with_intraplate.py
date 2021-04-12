#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crossflow heat exchanger model 
@author: afurlong
"""

import cantera as ct
import numpy as np
import math
from scipy.integrate import solve_ivp

class crossflow_hx(object):
    """
    Crossflow heat exchanger model for a single printed circuit heat exchanger (PCHE).
    
    Parameters
    -----
    reactant_in : list of four elements
        0. reactant composition, Cantera library of compositions, units of mol%
        1. mass flow rate through reactant plate, units of kg/s 
        2. reactant inlet temperature, units K        
        3. reactant inlet absolute pressure, units of Pa
    
    utility_in : list of four elements
        0. utility composition, Cantera library of compositions, units of mol%        
        1. mass flow rate through utility plate, units of kg/s         
        2. utility inlet temperature, units K        
        3. utility inlet absolute pressure, units of Pa
        
    fuel_in : list of four elements
        0. fuel composition, Cantera library of compositions, units of mol%
        1. mass flow rate through fuel plate, units of kg/s 
        2. fuel inlet temperature, units K        
        3. fuel inlet absolute pressure, units of Pa
        
    dims : list of five elements
        0. reactant channel diameter, units of m
        1. utility channel diameter, units of m
        2. number of reactant channels, dimensionless
        3. number of utility channels, dimensionless
        4. wall thickness between channels, units of m
        5. plate thickness, units of m
        
        note : fuel plate geometry assumed to be identical to reactant plate geometry
        
        
    Returns
    -----
    solve_hx method : Returns six arrays
        array0 : [reactant temperature profile, reactant pressure profile]
            reactant temperature profile: cell-by-cell array with inlet shown in 
            column 0, units of K
            
            reactant pressure profile: cell-by-cell array with inlet shown in 
            column 0, units of Pa
        array1 : [utility temperature profile, utility pressure profile]
            fuel temperature profile: cell-by-cell array with inlet shown in 
            row 0, units of K
            
            fuel pressure profile: cell-by-cell array with inlet shown in 
            row 0, units of Pa
        array2 : [fuel temperature profile, fuel pressure profile]
            fuel temperature profile: cell-by-cell array with inlet shown in 
            row 0, units of K
            
            fuel pressure profile: cell-by-cell array with inlet shown in 
            column 0, units of Pa
        array3 : reactant plate temperature profile: cell-by-cell array with inlet shown in 
            column 0, units of K
        array4 : utility plate temperature profile: cell-by-cell array with inlet shown in 
            row 0, units of K
        array5 : fuel plate temperature profile: cell-by-cell array with inlet shown in 
            column 0, units of K
            
            
    transientHX method : For use with an implicit ODE solver
    
    Reference
    -----
    See other methods in class.
    
    Applicability
    -----
    Applicable for laminar flow (Re < 2300).
    
    Not suitable for use beyond approximately ideal gas conditions.
            
    """
    
    def __init__(self, reactant_in, utility_in, fuel_in, dims):
        #list of [[stream composition], mass flow [kg/s], Temp [K], Pressure [Pa abs]]
        self.reactant_in = reactant_in
        self.utility_in = utility_in
        self.fuel_in = fuel_in
        #dimensions are [reactant channel diameter, utility channel diameter, number reactant channels, number utility channels, wall thicknesses for channels]
        self.dims = dims


        #cross-sectional area, hydraulic diameter, square root of cross section
        self.reactant_cs = math.pi*self.dims[0]**2/8
        self.utility_cs = math.pi*self.dims[1]**2/8
        self.reactant_dh = 4*self.reactant_cs/(math.pi*self.dims[0]/2 + self.dims[0])
        self.utility_dh = 4*self.utility_cs/(math.pi*self.dims[1]/2 + self.dims[1])
        self.reactant_sqrtA = self.reactant_cs**0.5
        self.utility_sqrtA = self.utility_cs**0.5
        
        self.reactant_eps = 0.5 #semicircular channel aspect ratio
        self.utility_eps = 0.5
        
        #revise this for more complex modelling?
        #self.hx_area = self.dims[0]*self.dims[1]
        #set heat exchange area for each case
        self.hx_area_uPuF = (math.pi*self.dims[1]/2)*(self.dims[0]+self.dims[4])
        self.hx_area_uPrF = self.dims[0]*(self.dims[1]+self.dims[4])
        self.hx_area_uPrP = (self.dims[0]+self.dims[4])*(self.dims[1]+self.dims[4])-self.hx_area_uPrF
        self.hx_area_rPrF = (math.pi*self.dims[0]/2)*(self.dims[1]+self.dims[4])
        self.hx_area_rPfF = self.dims[0]*(self.dims[1]+self.dims[4])
        self.hx_area_rPfP = (self.dims[0]+self.dims[4])*(self.dims[1]+self.dims[4])-self.hx_area_rPfF
        self.hx_area_fPfF = (math.pi*self.dims[0]/2)*(self.dims[1]+self.dims[4])
        
        #determine the volume of metal and set metal properties for metal temperature profile
        #volume = volume of total cell less area takem by the channel
        self.Vcell_utilityPlate = (self.dims[0]+self.dims[4])*(self.dims[1]+self.dims[4])*self.dims[5] - self.utility_cs*(self.dims[0]+self.dims[4])
        self.Vcell_reactantPlate = (self.dims[0]+self.dims[4])*(self.dims[1]+self.dims[4])*self.dims[5] - self.reactant_cs*(self.dims[1]+self.dims[4])
        self.Vcell_fuelplate = self.Vcell_reactantPlate
        self.Vcell_utility = self.utility_cs*(self.dims[0]+self.dims[4])
        self.Vcell_reactant = self.reactant_cs*(self.dims[1]+self.dims[4])
        self.Vcell_fuel = self.Vcell_reactant
        
        self.metalRho = 8000
        self.metalcp = 500
        self.metalk = 50
        
        #set up initial cantera fluids - gri30 good for combustion of natural gas
        #not setting up initial fluid conditions here
        self.reactant = ct.Solution('gri30.xml')
        self.utility = ct.Solution('gri30.xml')
        self.fuel = ct.Solution('gri30.xml')
        self.reactant.transport_model = 'Mix'
        self.utility.transport_model = 'Mix'
        self.fuel.transport_model = 'Mix'
        
        """
        set up inital arrays to reserve memory
        needed arrays are temperature, pressure, density, velocity, viscosity, 
        reynolds number, specific heat capacity, thermal conductivity, 
        prandtl number, nusselt number, convective heat transfer coefficient, 
        overall heat transfer coefficient, total heat transfer, z for hx calcs length
        """
        
        #create empty identical arrays for each variable - different dimensions for each fluid
        self.reactant_T, self.reactant_P, self.reactant_rho, self.reactant_u, self.reactant_mu, self.reactant_Re, self.reactant_cp, self.reactant_k, self.reactant_Pr, self.reactant_Nu, self.reactant_h = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*11)
        self.utility_T, self.utility_P, self.utility_rho, self.utility_u, self.utility_mu, self.utility_Re, self.utility_cp, self.utility_k, self.utility_Pr, self.utility_Nu, self.utility_h = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 2)]*11)
        self.fuel_T, self.fuel_P, self.fuel_rho, self.fuel_u, self.fuel_mu, self.fuel_Re, self.fuel_cp, self.fuel_k, self.fuel_Pr, self.fuel_Nu, self.fuel_h = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*11)
        
        #self.U, self.Q = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*2)
        self.Q_utilityPlate, self.Q_utilityFluid, self.Q_reactantsPlate, self.Q_reactantsFluid, self.Q_fuelPlate, self.Q_fuelFluid = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*6)
        self.T_reactantPlate, self.T_fuelPlate = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*2)
        self.T_utilityPlate = self.emptyarray(self.dims[2], self.dims[3], 2)
        #axial position
        self.reactant_z = np.linspace(start = 0, stop = self.dims[3]*(self.dims[1] + self.dims[4]) - self.dims[4], num = self.dims[3] + 1)
        self.utility_z = np.linspace(start = 0, stop = self.dims[2]*(self.dims[0] + self.dims[4]) - self.dims[4], num = self.dims[2] + 1)
        
        #setup for pressure drop correlations
        self.reactant_lplus, self.reactant_f = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*2)
        self.utility_lplus, self.utility_f = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 2)]*2)
        self.fuel_lplus, self.fuel_f = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*2)
        
    def emptyarray(self, dim1, dim2, direction):
        """
        Method for producing an empty array with either an additional row 
        or column for an initial condition.

        Parameters
        ----------
        dim1 : int
            Number of rows in the array
        dim2 : int
            Number of columns in the array
        direction : int with value 1 or 2
            Add an additional row (1) or column (2)

        Returns
        ----------
        empty : array
            An array of zeros with either an additional row or column

        """
        #take number of channels and if the array should have an extra row or column
        #extra column is for reactant channel initialization, and extra row is for utility channel
        if direction == 1:
            narrow = 1
            wide = 0
        elif direction == 2:
            narrow = 0
            wide = 1
        empty = np.zeros(shape = [dim1+1*wide, dim2+1*narrow], dtype = float)
        return(empty)
        
    def reynolds(self, density, velocity, char_len, viscosity):
        """
        Function to evaluate the Reynolds number for the given fluid and geometry

        Parameters
        ----------
        density : Float
            fluid density, units of kg m-3
        velocity : Float
            fluid velocity, units of m s-1
        char_len : Float
            characteristic dimension for fluid, units of m
        viscosity : Float
            fluid viscosity, units of kg m-1 s-1

        Returns
        -------
        re : Float
            Reynolds number, dimensionless

        """
        #basic reynolds number calculation
        re = density*velocity*char_len/viscosity
        return(re)

    def entryNu(self, z, Re, Pr, f, eps, sqrtA):
        """
        Function to evaluate the Nusselt number for developing laminar flow.

        Parameters
        ----------
        z : Float
            axial position, units of m
        Re : Float
            Reynolds number, dimensionless
        Pr : Float
            Prandtl number, dimensionless
        f : Float
            Laminar flow developing region friction factor, dimensionless. 
            Given by devflow_friction method.
        eps : Float
            Aspect ratio, dimensionless
        sqrtA : Float
            Sqrt flow area, units of m

        Returns
        -------
        Nu : Float
            Nusselt number for the location in the heat exchanger, dimensionless

        Reference
        ------
        Muzychka, Y. S., & Yovanovich, M. M. (2004). Laminar Forced Convection 
        Heat Transfer in the Combined Entry Region of Non-Circular Ducts. 
        Journal of Heat Transfer, 126(1), 54–61. 
        https://doi.org/10.1115/1.1643752

        
        Applicability
        --------
        Applicanble for laminar flow in the developing region. Approximately 
        suitable for fully developed laminar flow.
        """
        #Muzychka & Yovanovich
        C1 = 3.24
        C2 = 1
        C3 = 0.409
        C4 = 1
        n = 5
        gamma = -0.1 #range is ~-0.3 to ~0.1, taking midpoint
        
        if z == 0:
            zstar = 0.0001 #refine this later
        else:
            zstar = z/(sqrtA*Re*Pr)
            
        Nu = (((C2*C3*(f*Re/zstar)**(1/3))**n + (C1*(f*Re/(8*math.pi**0.5*eps**gamma)))**n))**(1/n)
        return(Nu)

    def lplus(self, L, ReSqrtA, sqrtA):
        """
        Function to evaluate the dimensionless duct length

        Parameters
        ----------
        L : Float
            Axial postion, units of m
        ReSqrtA : Float
            Reynolds number with sqrt channel area as the characteristic 
            dimension, units of m
        sqrtA : Float
            Sqrt of channel flow area, units of m

        Returns
        -------
        lplus : Float
            Dimensionless duct length, dimensionless
            
        Reference
        -------
        Muzychka, Y. S., & Yovanovich, M. M. (2009). Pressure Drop in Laminar 
        Developing Flow in Noncircular Ducts: A Scaling and Modeling Approach. 
        Journal of Fluids Engineering, 131(11). 
        https://doi.org/10.1115/1.4000377


        """
        #Muzychka & Yovanovich
        lplus = L/(sqrtA*ReSqrtA)
        return(lplus)
        
    def devflow_friction(self, lplus, eps, ReSqrtA):
        """
        Evaluate the friction factor for developing laminar flow

        Parameters
        ----------
        lplus : Float
            Dimensionless duct length, dimensionless. Given by lplus method.
        eps : Float
            Aspect ratio, dimensionless.
        ReSqrtA : Float
            Reynolds number with characteristic length of sqrt flow area.
            Dimensionless.

        Returns
        -------
        f : Float
            Fanning friction factor, dimensionless.
            
        Reference
        -----
        Muzychka, Y. S., & Yovanovich, M. M. (2009). Pressure Drop in Laminar 
        Developing Flow in Noncircular Ducts: A Scaling and Modeling Approach. 
        Journal of Fluids Engineering, 131(11). 
        https://doi.org/10.1115/1.4000377

        Applicability
        -----
        Applicanble for laminar flow in the developing region. Approximately 
        suitable for fully developed laminar flow.

        """
        #developing flow friction factor - give a minimum length to prevent divide by 0 errors
        n = 1.97 #1.97 - 2 seem to be the right range for a semicircle
        if lplus == 0:
            lplus = 1 #refine this later
        f = (((3.44/(lplus**0.5))**n + (12/((eps**0.5) * (1 + eps) * (1 - 192*eps/(math.pi**5)*np.tanh(math.pi/(2*eps)))))**n)**(1/n))/(ReSqrtA)
        return(f)
    
    def fluid_properties_reactant(self, T, P):
        """
        Use Cantera to evaluate fluid properties for given conditions. 
        Uses the reactant gas object to reduce composition updates.

        Parameters
        ----------
        T : Float
            reactant temperature, units of K.
        P : Float
            reactant pressure, units of Pa.

        Returns
        -------
        rho : Float
            reactant density, units of kg m-3
        mu : Float
            reactant viscosity, units of kg m-1 s-1
        cp : Float
            reactant specific heat capacity, mass basis, units of J kg-1 K-1
        k : Float
            reactant thermal conducivity, units of W m-1 K-1
        Pr : Float
            reactant Prandtl number, dimensionless
            
        Applicability
        -----
        Applicable for near-ideal conditions. Cantera using GRI-Mech 3.0 which
        is an ideal-gas solver.

        """
        self.reactant.TP = T, P                
        rho = self.reactant.density
        mu = self.reactant.viscosity
        cp = self.reactant.cp_mass
        k = self.reactant.thermal_conductivity
        Pr = mu*cp/k
        return(rho, mu, cp, k, Pr)
    
    def fluid_properties_utility(self, T, P):
        """
        Use Cantera to evaluate fluid properties for given conditions. 
        Uses the utility gas object to reduce composition updates.

        Parameters
        ----------
        T : Float
            Utility temperature, units of K.
        P : Float
            Utility pressure, units of Pa.

        Returns
        -------
        rho : Float
            Utility density, units of kg m-3
        mu : Float
            Utility viscosity, units of kg m-1 s-1
        cp : Float
            Utility specific heat capacity, mass basis, units of J kg-1 K-1
        k : Float
            Utility thermal conducivity, units of W m-1 K-1
        Pr : Float
            Utility Prandtl number, dimensionless
            
        Applicability
        -----
        Applicable for near-ideal conditions. Cantera using GRI-Mech 3.0 which
        is an ideal-gas solver.

        """
        self.utility.TP = T, P
        rho = self.utility.density
        mu = self.utility.viscosity
        cp = self.utility.cp_mass
        k = self.utility.thermal_conductivity
        Pr = mu*cp/k
        return(rho, mu, cp, k, Pr)
    
    def fluid_properties_fuel(self, T, P):
        """
        Use Cantera to evaluate fluid properties for given conditions. 
        Uses the fuel gas object to reduce composition updates.

        Parameters
        ----------
        T : Float
            reactant temperature, units of K.
        P : Float
            reactant pressure, units of Pa.

        Returns
        -------
        rho : Float
            reactant density, units of kg m-3
        mu : Float
            reactant viscosity, units of kg m-1 s-1
        cp : Float
            reactant specific heat capacity, mass basis, units of J kg-1 K-1
        k : Float
            reactant thermal conducivity, units of W m-1 K-1
        Pr : Float
            reactant Prandtl number, dimensionless
            
        Applicability
        -----
        Applicable for near-ideal conditions. Cantera using GRI-Mech 3.0 which
        is an ideal-gas solver.

        """
        self.fuel.TP = T, P                
        rho = self.fuel.density
        mu = self.fuel.viscosity
        cp = self.fuel.cp_mass
        k = self.fuel.thermal_conductivity
        Pr = mu*cp/k
        return(rho, mu, cp, k, Pr)
    
    def update_properties(self, T_reactant, T_utility, T_fuel):
        """
        Update arrays of properties and convective heat transfer correlations 
        for new temperature profiles. For use in the transient solver. 
        Assuming pressures are constant from the steady-state model.

        Parameters
        ----------
        T_reactant : Array
            reactant temperatures, float, units of K.
        T_utility : Array
            Utility temperatures, float, units of K.

        Returns
        -------
        None.

        """
        for j in range(self.dims[3]):
            for i in range(self.dims[2]):
                self.reactant_rho[i, j], self.reactant_mu[i, j], self.reactant_cp[i, j],self.reactant_k[i, j], self.reactant_Pr[i, j] = self.fluid_properties_reactant(T_reactant[i, j], self.reactant_P[i, j])
                self.utility_rho[i, j], self.utility_mu[i, j], self.utility_cp[i, j],self.utility_k[i, j], self.utility_Pr[i, j] = self.fluid_properties_utility(T_utility[i, j], self.utility_P[i, j])
                self.fuel_rho[i, j], self.fuel_mu[i, j], self.fuel_cp[i, j],self.fuel_k[i, j], self.fuel_Pr[i, j] = self.fluid_properties_fuel(T_fuel[i, j], self.fuel_P[i, j])

                self.reactant_u[i, j] = self.reactant_u0*self.reactant_rho0/self.reactant_rho[i, j]
                self.utility_u[i, j] = self.utility_u0*self.utility_rho0/self.utility_rho[i, j]
                self.fuel_u[i, j] = self.fuel_u0*self.fuel_rho0/self.fuel_rho[i, j]

                self.reactant_Re[i, j] = self.reynolds(self.reactant_rho[i, j], self.reactant_u[i, j], self.reactant_sqrtA, self.reactant_mu[i, j])
                self.utility_Re[i, j] = self.reynolds(self.utility_rho[i, j], self.utility_u[i, j], self.utility_sqrtA, self.utility_mu[i, j])
                self.fuel_Re[i, j] = self.reynolds(self.fuel_rho[i, j], self.fuel_u[i, j], self.reactant_sqrtA, self.fuel_mu[i, j])

                
                if self.reactant_Re[i, j] < 2300: 
                    self.reactant_Nu[i, j] = self.entryNu(self.reactant_z[j], self.reactant_Re[i, j], self.reactant_Pr[i, j], self.reactant_f[i, j], self.reactant_eps, self.reactant_sqrtA)
                else:
                    self.reactant_Nu[i, j] = 0.023*self.reactant_Re[i, j]**0.8*self.reactant_Pr[i, j]**0.3
                    
                if self.utility_Re[i, j] < 2300:
                    self.utility_Nu[i, j] = self.entryNu(self.utility_z[i], self.utility_Re[i, j], self.utility_Pr[i, j], self.utility_f[i, j], self.utility_eps, self.utility_sqrtA)
                else:
                    self.utility_Nu[i, j] = 0.023*self.utility_Re[i, j]**0.8*self.utility_Pr[i, j]**0.4
                
                if self.fuel_Re[i, j] < 2300: 
                    self.fuel_Nu[i, j] = self.entryNu(self.reactant_z[j], self.fuel_Re[i, j], self.fuel_Pr[i, j], self.fuel_f[i, j], self.reactant_eps, self.reactant_sqrtA)
                else:
                    self.fuel_Nu[i, j] = 0.023*self.fuel_Re[i, j]**0.8*self.fuel_Pr[i, j]**0.3
                  
                
                self.reactant_h[i, j] = self.reactant_Nu[i, j]*self.reactant_k[i, j]/self.reactant_sqrtA
                self.utility_h[i, j] = self.utility_Nu[i, j]*self.utility_k[i, j]/self.utility_sqrtA
                self.fuel_h[i, j] = self.fuel_Nu[i, j]*self.fuel_k[i, j]/self.reactant_sqrtA

                
    def unwrap_T(self, T_vector):
        """
        Used to manipulate vector of reactant and utility temperatures profiles
        into two arrays. Manipulates data into easily iterable form after 
        passing through the 1-dimensional ODE solver.

        Parameters
        ----------
        T_vector : List
            Temperature profile produced by ODE solver, units of K.

        Returns
        -------
        initial_reactant_Temps : Array
            2-Dimensional reactant plate temperature profile, with column 0 as the 
            inlet. Units of K.
        initial_utility_Temps : Array
            2-Dimensional utility plate temperature profile, with row 0 as the 
            inlet. Units of K.

        """
        initial_reactant_Temps = T_vector[0:int(self.dims[2]*self.dims[3])].reshape(int(self.dims[2]), int(self.dims[3]))
        initial_utility_Temps = T_vector[int(self.dims[2]*self.dims[3]):2*int(self.dims[2]*self.dims[3])].reshape(int(self.dims[2]), int(self.dims[3]))
        initial_fuel_Temps = T_vector[2*int(self.dims[2]*self.dims[3]):3*int(self.dims[2]*self.dims[3])].reshape(int(self.dims[2]), int(self.dims[3]))
        initial_reactantPlate_Temps = T_vector[3*int(self.dims[2]*self.dims[3]):4*int(self.dims[2]*self.dims[3])].reshape(int(self.dims[2]), int(self.dims[3]))
        initial_utilityPlate_Temps = T_vector[4*int(self.dims[2]*self.dims[3]):5*int(self.dims[2]*self.dims[3])].reshape(int(self.dims[2]), int(self.dims[3]))
        initial_fuelPlate_Temps = T_vector[5*int(self.dims[2]*self.dims[3]):6*int(self.dims[2]*self.dims[3])].reshape(int(self.dims[2]), int(self.dims[3]))
        
        return initial_reactant_Temps, initial_utility_Temps, initial_fuel_Temps, initial_reactantPlate_Temps, initial_utilityPlate_Temps, initial_fuelPlate_Temps
        
    def solvehx(self):
        """
        Method to solve steady-state temperature and pressure profiles for reactant
        and utility channels. Only solves when called. 

        Returns
        -------
        reactantout : Array
            element1: Temperature profile for reactant plate, with inlet in column 
            0 and outlet in column n. Units of K.
            
            element2: Pressure profile for reactant plate, with inlet in column 0 
            and outlet in column n. Units of Pa abs.
        
        utilityout : Array
            element1: Temperature profile for utility plate, with inlet in 
            row 0 and outlet in row n. Units of K.
            
            element2: Pressure profile for utility plate, with inlet in row 0 
            and outlet in row n. Units of Pa abs.
                
        """
        #set initial fluid conditions and arrays
        self.reactant.TPY = self.reactant_in[2], self.reactant_in[3], self.reactant_in[0]
        self.utility.TPY = self.utility_in[2], self.utility_in[3], self.utility_in[0]
        self.fuel.TPY = self.fuel_in[2], self.fuel_in[3], self.fuel_in[0]
        
        #set boundary conditions
        self.reactant_T[:, 0] = self.reactant_in[2]
        self.reactant_P[:, 0] = self.reactant_in[3]
        self.utility_T[0, :] = self.utility_in[2]
        self.utility_P[0, :] = self.utility_in[3]
        self.fuel_T[:, 0] = self.fuel_in[2]
        self.fuel_P[:, 0] = self.fuel_in[3]
        
        #set initial estimates for temperatures in the solid phase using arithmatic means
        #self.T_utilityPlate[0, :] = (self.utility_in[2] + self.reactant_in[2])/2
        #self.T_reactantPlate[:, 0] = (self.reactant_in[2] + self.fuel_in[2])/2
        #self.T_fuelPlate[:, 0] = (self.reactant_in[2] + self.fuel_in[2])/2 #assuming mirrored structure
        self.T_utilityPlate[0, :] = (self.utility_in[2]*self.utility_in[1] + self.reactant_in[2]*self.reactant_in[1])/(self.utility_in[1]+self.reactant_in[1])
        self.T_reactantPlate[:, 0] = self.reactant_in[2]
        #self.T_fuelPlate[:, 0] = (self.reactant_in[2]*self.reactant_in[1] + self.fuel_in[2]*self.fuel_in[1])/(self.reactant_in[1]+self.fuel_in[1])
        self.T_fuelPlate[:, 0] = self.reactant_in[2]
        
        #use cantera to determine fluid properties
        self.reactant_rho[:, 0], self.reactant_mu[:, 0], self.reactant_cp[:, 0], self.reactant_k[:, 0], self.reactant_Pr[:, 0] = self.fluid_properties_reactant(self.reactant_in[2], self.reactant_in[3])
        self.utility_rho[0, :], self.utility_mu[0, :], self.utility_cp[0, :], self.utility_k[0, :], self.utility_Pr[0, :] = self.fluid_properties_utility(self.utility_in[2], self.utility_in[3])
        self.fuel_rho[:, 0], self.fuel_mu[:, 0], self.fuel_cp[:, 0], self.fuel_k[:, 0], self.fuel_Pr[:, 0] = self.fluid_properties_fuel(self.fuel_in[2], self.fuel_in[3])

        #inlet velocity = mass flow rate in / density / # channels / channel cross section
        self.reactant_u[:, 0] = self.reactant_in[1]/(self.reactant_rho[0, 0]*self.dims[2]*self.reactant_cs)
        self.utility_u[0, :] = self.utility_in[1]/(self.utility.density*self.dims[3]*self.utility_cs)
        self.fuel_u[:, 0] = self.fuel_in[1]/(self.fuel_rho[0, 0]*self.dims[2]*self.reactant_cs)

        #grab initial velocity and density to determine later velocity via continuity equation
        self.reactant_u0 = self.reactant_u[0, 0]    
        self.utility_u0 = self.utility_u[0, 0]
        self.fuel_u0 = self.fuel_u[0, 0]    
        self.reactant_rho0 = self.reactant_rho[0, 0]
        self.utility_rho0 = self.utility_rho[0, 0]
        self.fuel_rho0 = self.fuel_rho[0, 0]

        #maybe update this soon?
        self.reactant_Re[:, 0] = self.reynolds(self.reactant_rho[0, 0], self.reactant_u[0, 0], self.reactant_sqrtA, self.reactant_mu[0, 0])
        self.utility_Re[0, :] = self.reynolds(self.utility_rho[0, 0], self.utility_u[0, 0], self.utility_sqrtA, self.utility_mu[0, 0])
        self.fuel_Re[:, 0] = self.reynolds(self.fuel_rho[0, 0], self.fuel_u[0, 0], self.reactant_sqrtA, self.fuel_mu[0, 0])
        
        #Muzychka & Yovanovich - entry region friction factor
        #L+ is 0 for the entry region
        self.reactant_f[:, 0] = self.devflow_friction(self.reactant_lplus[0, 0], self.reactant_eps, self.reactant_Re[0, 0])
        self.utility_f[0, :] = self.devflow_friction(self.utility_lplus[0, 0], self.utility_eps, self.utility_Re[0, 0])
        self.fuel_f[:, 0] = self.devflow_friction(self.fuel_lplus[0, 0], self.reactant_eps, self.fuel_Re[0, 0])

        
        #Muzychka & Yovanovich - combined entry region for small z*
        self.reactant_Nu[:, 0] = self.entryNu(0, self.reactant_Re[0, 0], self.reactant_Pr[0, 0], self.reactant_f[0, 0], self.reactant_eps, self.reactant_sqrtA)
        self.utility_Nu[0, :] = self.entryNu(0, self.utility_Re[0, 0], self.utility_Pr[0, 0], self.utility_f[0, 0], self.utility_eps, self.utility_sqrtA)
        self.fuel_Nu[:, 0] = self.entryNu(0, self.fuel_Re[0, 0], self.fuel_Pr[0, 0], self.fuel_f[0, 0], self.reactant_eps, self.reactant_sqrtA)
        
        #should this be sqrtA or dh?
        self.reactant_h[:, 0] = self.reactant_Nu[0, 0] * self.reactant_k[0, 0]/self.reactant_sqrtA
        self.utility_h[0, :] = self.utility_Nu[0, 0] * self.utility_k[0, 0]/self.utility_sqrtA
        self.fuel_h[:, 0] = self.fuel_Nu[0, 0] * self.fuel_k[0, 0]/self.reactant_sqrtA
        
        #now iterate through and solve for everything
        for j in range(0, self.dims[3]):
            for i in range(0, self.dims[2]):
                #calculate heat tranfer for all interactions
                self.Q_utilityFluid[i, j] = 0.25*(self.utility_h[i, j]*self.hx_area_uPuF*(self.T_utilityPlate[i, j] - self.utility_T[i, j]))
                self.Q_utilityPlate[i, j] = 0.25*(-self.Q_utilityFluid[i, j] + self.metalk*self.hx_area_uPrP*(self.T_reactantPlate[i, j] - self.T_utilityPlate[i, j])/self.dims[5] + self.reactant_h[i, j]*self.hx_area_uPrF*(self.reactant_T[i, j] - self.T_utilityPlate[i, j]))
                self.Q_reactantsFluid[i, j] = 0.25*(-self.reactant_h[i, j]*self.hx_area_uPrF*(self.reactant_T[i, j] - self.T_utilityPlate[i, j]) - self.reactant_h[i, j]*self.hx_area_rPrF*(self.reactant_T[i, j] - self.T_reactantPlate[i, j]))
                self.Q_reactantsPlate[i, j] = 0.25*(self.reactant_h[i, j]*self.hx_area_rPrF*(self.reactant_T[i, j] - self.T_reactantPlate[i, j]) - self.fuel_h[i, j]*self.hx_area_rPfF*(self.T_reactantPlate[i, j] - self.fuel_T[i, j]) - self.metalk*self.hx_area_rPfP*(self.T_reactantPlate[i, j] - self.T_fuelPlate[i, j])/self.dims[5] - self.metalk*self.hx_area_uPrP*(self.T_reactantPlate[i, j] - self.T_utilityPlate[i, j])/self.dims[5])
                self.Q_fuelFluid[i, j] = 0.25*(self.fuel_h[i, j]*self.hx_area_rPfF*(self.T_reactantPlate[i, j] - self.fuel_T[i, j]) - self.fuel_h[i, j]*self.hx_area_fPfF*(self.fuel_T[i, j] - self.T_fuelPlate[i, j]))
                self.Q_fuelPlate[i, j] = 0.25*(self.fuel_h[i, j]*self.hx_area_fPfF*(self.fuel_T[i, j] - self.T_fuelPlate[i, j]) + self.metalk*self.hx_area_rPfP*(self.T_reactantPlate[i, j] - self.T_fuelPlate[i, j])/self.dims[5])
                #print(self.Q_utilityFluid[i, j] + self.Q_utilityPlate[i, j] + self.Q_reactantsFluid[i, j] + self.Q_reactantsPlate[i, j] + self.Q_fuelFluid[i, j] + self.Q_fuelPlate[i, j])
                #totalQ = self.Q_utilityFluid[i, j] + self.Q_utilityPlate[i, j] + self.Q_reactantsFluid[i, j] + self.Q_reactantsPlate[i, j] + self.Q_fuelFluid[i, j] + self.Q_fuelPlate[i, j]
                #print(totalQ)
                
                #use the mass flow rate through each channel instead of the velocity in each channel for the heat transferred
                #self.reactant_T[i, j+1] = self.reactant_T[i, j] - self.Q[i, j]/(self.reactant_in[1]/self.dims[2]*self.reactant_cp[i, j])
                #self.utility_T[i+1, j] = self.utility_T[i, j] + self.Q[i, j]/(self.utility_in[1]/self.dims[3]*self.utility_cp[i, j])
                self.utility_T[i+1, j] = self.utility_T[i, j] + self.Q_utilityFluid[i, j]/(self.utility_in[1]/self.dims[3]*self.utility_cp[i, j])
                self.reactant_T[i, j+1] = self.reactant_T[i, j] + self.Q_reactantsFluid[i, j]/(self.reactant_in[1]/self.dims[2]*self.reactant_cp[i, j])
                self.fuel_T[i, j+1] = self.fuel_T[i, j] + self.Q_fuelFluid[i, j]/(self.fuel_in[1]/self.dims[2]*self.fuel_cp[i, j])
                
                self.T_fuelPlate[i, j+1] = self.T_fuelPlate[i, j] + self.Q_fuelPlate[i, j]/(self.metalRho*self.metalcp*self.Vcell_fuelplate)
                self.T_utilityPlate[i+1, j] = self.T_utilityPlate[i, j] + self.Q_utilityPlate[i, j]/(self.metalRho*self.metalcp*self.Vcell_utilityPlate)
                self.T_reactantPlate[i, j+1] = self.T_reactantPlate[i, j] + self.Q_reactantsPlate[i, j]/(self.metalRho*self.metalcp*self.Vcell_reactantPlate)
                
                #assuming incompressible flow given low velocity
                self.reactant_P[i, j+1] = self.reactant_P[i, j] - 2*self.reactant_f[i, j] * (self.dims[1] + self.dims[4]) / self.reactant_dh *self.reactant_rho[i, j]*self.reactant_u[i, j]**2
                self.utility_P[i+1, j] = self.utility_P[i, j] - 2*self.utility_f[i, j] * (self.dims[0] + self.dims[4]) / self.utility_dh *self.utility_rho[i, j]*self.utility_u[i, j]**2
                self.fuel_P[i, j+1] = self.fuel_P[i, j] - 2*self.fuel_f[i, j] * (self.dims[1] + self.dims[4]) / self.reactant_dh *self.fuel_rho[i, j]*self.fuel_u[i, j]**2

        
                #update thermophysical and transport properties for new conditions
                self.reactant_rho[i, j+1], self.reactant_mu[i, j+1], self.reactant_cp[i, j+1],self.reactant_k[i, j+1], self.reactant_Pr[i, j+1] = self.fluid_properties_reactant(self.reactant_T[i, j+1], self.reactant_P[i, j+1])
                self.utility_rho[i+1, j], self.utility_mu[i+1, j], self.utility_cp[i+1, j],self.utility_k[i+1, j], self.utility_Pr[i+1, j] = self.fluid_properties_utility(self.utility_T[i+1, j], self.utility_P[i+1, j])
                self.fuel_rho[i, j+1], self.fuel_mu[i, j+1], self.fuel_cp[i, j+1],self.fuel_k[i, j+1], self.fuel_Pr[i, j+1] = self.fluid_properties_fuel(self.fuel_T[i, j+1], self.fuel_P[i, j+1])

                
                #update velocity
                self.reactant_u[i, j+1] = self.reactant_u[i, j]*self.reactant_rho[i, j]/self.reactant_rho[i, j+1]
                self.utility_u[i+1, j] = self.utility_u[i, j]*self.utility_rho[i, j]/self.utility_rho[i+1, j]
                self.fuel_u[i, j+1] = self.fuel_u[i, j]*self.fuel_rho[i, j]/self.fuel_rho[i, j+1]
                
                self.reactant_Re[i, j+1] = self.reynolds(self.reactant_rho[i, j+1], self.reactant_u[i, j+1], self.reactant_sqrtA, self.reactant_mu[i, j+1])
                self.utility_Re[i+1, j] = self.reynolds(self.utility_rho[i+1, j], self.utility_u[i+1, j], self.utility_sqrtA, self.utility_mu[i+1, j])
                self.fuel_Re[i, j+1] = self.reynolds(self.fuel_rho[i, j+1], self.fuel_u[i, j+1], self.reactant_sqrtA, self.fuel_mu[i, j+1])
                
                #update lplus and friction factors
                self.reactant_lplus[i, j+1] = self.lplus(self.reactant_z[j+1], self.reactant_Re[i, j+1], self.reactant_sqrtA)
                self.utility_lplus[i+1, j] = self.lplus(self.utility_z[i+1], self.utility_Re[i+1, j], self.utility_sqrtA)
                self.fuel_lplus[i, j+1] = self.lplus(self.reactant_z[j+1], self.fuel_Re[i, j+1], self.reactant_sqrtA)
                self.reactant_f[i, j+1] = self.devflow_friction(self.reactant_lplus[i, j+1], self.reactant_eps, self.reactant_Re[i, j+1])
                self.utility_f[i+1, j] = self.devflow_friction(self.utility_lplus[i+1, j], self.utility_eps, self.utility_Re[i+1, j])
                self.fuel_f[i, j+1] = self.devflow_friction(self.fuel_lplus[i, j+1], self.reactant_eps, self.fuel_Re[i, j+1])
                
                #determine developing region heat transfer coefficients
                if self.reactant_Re[i, j+1] < 2300: 
                    self.reactant_Nu[i, j+1] = self.entryNu(self.reactant_z[j+1], self.reactant_Re[i, j+1], self.reactant_Pr[i, j+1], self.reactant_f[i, j+1], self.reactant_eps, self.reactant_sqrtA)
                else:
                    self.reactant_Nu[i, j+1] = 0.023*self.reactant_Re[i, j+1]**0.8*self.reactant_Pr[i, j+1]**0.3
                    
                if self.utility_Re[i+1, j] < 2300:
                    self.utility_Nu[i+1, j] = self.entryNu(self.utility_z[i+1], self.utility_Re[i+1, j], self.utility_Pr[i+1, j], self.utility_f[i+1, j], self.utility_eps, self.utility_sqrtA)
                else:
                    self.utility_Nu[i+1, j] = 0.023*self.utility_Re[i+1, j]**0.8*self.utility_Pr[i+1, j]**0.4

                if self.fuel_Re[i, j+1] < 2300: 
                    self.fuel_Nu[i, j+1] = self.entryNu(self.reactant_z[j+1], self.fuel_Re[i, j+1], self.fuel_Pr[i, j+1], self.fuel_f[i, j+1], self.reactant_eps, self.reactant_sqrtA)
                else:
                    self.fuel_Nu[i, j+1] = 0.023*self.fuel_Re[i, j+1]**0.8*self.fuel_Pr[i, j+1]**0.4
                    
                self.reactant_h[i, j+1] = self.reactant_Nu[i, j+1]*self.reactant_k[i, j+1]/self.reactant_sqrtA
                self.utility_h[i+1, j] = self.utility_Nu[i+1, j]*self.utility_k[i+1, j]/self.utility_sqrtA
                self.fuel_h[i, j+1] = self.fuel_Nu[i, j+1]*self.fuel_k[i, j+1]/self.reactant_sqrtA
                #print(i, j, self.T_utilityPlate[i, j], self.fuel_P[i, j])
                
                #print(i, j, self.utility_T[i, j], self.T_utilityPlate[i, j], self.reactant_T[i, j], self.T_reactantPlate[i, j])

        for n in range(0):
            for j in range(1, self.dims[3]):
                    for i in range(1, self.dims[2]):
                        #calculate heat tranfer for all interactions
                        self.Q_utilityFluid[i, j] = 0.2*(self.utility_h[i, j]*self.hx_area_uPuF*(self.T_utilityPlate[i, j] - self.utility_T[i, j]))
                        self.Q_utilityPlate[i, j] = 0.2*(-self.Q_utilityFluid[i, j] + self.metalk*self.hx_area_uPrP*(self.T_reactantPlate[i, j] - self.T_utilityPlate[i, j])/self.dims[5] + self.reactant_h[i, j]*self.hx_area_uPrF*(self.reactant_T[i, j] - self.T_utilityPlate[i, j]))
                        self.Q_reactantsFluid[i, j] = 0.2*(-self.reactant_h[i, j]*self.hx_area_uPrF*(self.reactant_T[i, j] - self.T_utilityPlate[i, j]) - self.reactant_h[i, j]*self.hx_area_rPrF*(self.reactant_T[i, j] - self.T_reactantPlate[i, j]))
                        self.Q_reactantsPlate[i, j] = 0.2*(self.reactant_h[i, j]*self.hx_area_rPrF*(self.reactant_T[i, j] - self.T_reactantPlate[i, j]) - self.fuel_h[i, j]*self.hx_area_rPfF*(self.T_reactantPlate[i, j] - self.fuel_T[i, j]) - self.metalk*self.hx_area_rPfP*(self.T_reactantPlate[i, j] - self.T_fuelPlate[i, j])/self.dims[5] - self.metalk*self.hx_area_uPrP*(self.T_reactantPlate[i, j] - self.T_utilityPlate[i, j])/self.dims[5])
                        self.Q_fuelFluid[i, j] = 0.2*(self.fuel_h[i, j]*self.hx_area_rPfF*(self.T_reactantPlate[i, j] - self.fuel_T[i, j]) - self.fuel_h[i, j]*self.hx_area_fPfF*(self.fuel_T[i, j] - self.T_fuelPlate[i, j]))
                        self.Q_fuelPlate[i, j] = 0.2*(self.fuel_h[i, j]*self.hx_area_fPfF*(self.fuel_T[i, j] - self.T_fuelPlate[i, j]) + self.metalk*self.hx_area_rPfP*(self.T_reactantPlate[i, j] - self.T_fuelPlate[i, j])/self.dims[5])
                        
                        #totalQ = self.Q_utilityFluid[i, j] + self.Q_utilityPlate[i, j] + self.Q_reactantsFluid[i, j] + self.Q_reactantsPlate[i, j] + self.Q_fuelFluid[i, j] + self.Q_fuelPlate[i, j]
                        #print(totalQ)
                        
                        #use the mass flow rate through each channel instead of the velocity in each channel for the heat transferred
                        self.utility_T[i, j] = self.utility_T[i, j] + self.Q_utilityFluid[i, j]/(self.utility_in[1]/self.dims[3]*self.utility_cp[i, j])
                        self.reactant_T[i, j] = self.reactant_T[i, j] + self.Q_reactantsFluid[i, j]/(self.reactant_in[1]/self.dims[2]*self.reactant_cp[i, j])
                        self.fuel_T[i, j] = self.fuel_T[i, j] + self.Q_fuelFluid[i, j]/(self.fuel_in[1]/self.dims[2]*self.fuel_cp[i, j])
                        
                        self.T_fuelPlate[i, j] = self.T_fuelPlate[i, j] + self.Q_fuelPlate[i, j]/(self.metalRho*self.metalcp*self.Vcell_fuelplate)
                        self.T_utilityPlate[i, j] = self.T_utilityPlate[i, j] + self.Q_utilityPlate[i, j]/(self.metalRho*self.metalcp*self.Vcell_utilityPlate)
                        self.T_reactantPlate[i, j] = self.T_reactantPlate[i, j] + self.Q_reactantsPlate[i, j]/(self.metalRho*self.metalcp*self.Vcell_reactantPlate)
                        
                        #assuming incompressible flow given low velocity
                        self.reactant_P[i, j] = self.reactant_P[i, j] - 2*self.reactant_f[i, j] * (self.dims[1] + self.dims[4]) / self.reactant_dh *self.reactant_rho[i, j]*self.reactant_u[i, j]**2
                        self.utility_P[i, j] = self.utility_P[i, j] - 2*self.utility_f[i, j] * (self.dims[0] + self.dims[4]) / self.utility_dh *self.utility_rho[i, j]*self.utility_u[i, j]**2
                        self.fuel_P[i, j] = self.fuel_P[i, j] - 2*self.fuel_f[i, j] * (self.dims[1] + self.dims[4]) / self.reactant_dh *self.fuel_rho[i, j]*self.fuel_u[i, j]**2
        
                
                        #update thermophysical and transport properties for new conditions
                        self.reactant_rho[i, j], self.reactant_mu[i, j], self.reactant_cp[i, j],self.reactant_k[i, j], self.reactant_Pr[i, j] = self.fluid_properties_reactant(self.reactant_T[i, j], self.reactant_P[i, j])
                        self.utility_rho[i, j], self.utility_mu[i, j], self.utility_cp[i, j],self.utility_k[i, j], self.utility_Pr[i, j] = self.fluid_properties_utility(self.utility_T[i, j], self.utility_P[i, j])
                        self.fuel_rho[i, j], self.fuel_mu[i, j], self.fuel_cp[i, j],self.fuel_k[i, j], self.fuel_Pr[i, j] = self.fluid_properties_fuel(self.fuel_T[i, j], self.fuel_P[i, j])
        
                        
                        #update velocity
                        self.reactant_u[i, j] = self.reactant_u0*self.reactant_rho[i, j]/self.reactant_rho0
                        self.utility_u[i, j] = self.utility_u0*self.utility_rho[i, j]/self.utility_rho0
                        self.fuel_u[i, j] = self.fuel_u0*self.fuel_rho[i, j]/self.fuel_rho0
                        
                        self.reactant_Re[i, j] = self.reynolds(self.reactant_rho[i, j], self.reactant_u[i, j], self.reactant_sqrtA, self.reactant_mu[i, j])
                        self.utility_Re[i, j] = self.reynolds(self.utility_rho[i, j], self.utility_u[i, j], self.utility_sqrtA, self.utility_mu[i, j])
                        self.fuel_Re[i, j] = self.reynolds(self.fuel_rho[i, j], self.fuel_u[i, j], self.reactant_sqrtA, self.fuel_mu[i, j])
                        
                        #update lplus and friction factors
                        self.reactant_lplus[i, j] = self.lplus(self.reactant_z[j], self.reactant_Re[i, j], self.reactant_sqrtA)
                        self.utility_lplus[i, j] = self.lplus(self.utility_z[i], self.utility_Re[i, j], self.utility_sqrtA)
                        self.fuel_lplus[i, j] = self.lplus(self.reactant_z[j], self.fuel_Re[i, j], self.reactant_sqrtA)
                        self.reactant_f[i, j] = self.devflow_friction(self.reactant_lplus[i, j], self.reactant_eps, self.reactant_Re[i, j])
                        self.utility_f[i, j] = self.devflow_friction(self.utility_lplus[i, j], self.utility_eps, self.utility_Re[i, j])
                        self.fuel_f[i, j] = self.devflow_friction(self.fuel_lplus[i, j], self.reactant_eps, self.fuel_Re[i, j])
                        
                        
                        #determine developing region heat transfer coefficients
                        if self.reactant_Re[i, j] < 2300: 
                            self.reactant_Nu[i, j] = self.entryNu(self.reactant_z[j], self.reactant_Re[i, j], self.reactant_Pr[i, j], self.reactant_f[i, j], self.reactant_eps, self.reactant_sqrtA)
                        else:
                            self.reactant_Nu[i, j] = 0.023*self.reactant_Re[i, j]**0.8*self.reactant_Pr[i, j]**0.3
                            
                        if self.utility_Re[i, j] < 2300:
                            self.utility_Nu[i, j] = self.entryNu(self.utility_z[i], self.utility_Re[i, j], self.utility_Pr[i, j], self.utility_f[i, j], self.utility_eps, self.utility_sqrtA)
                        else:
                            self.utility_Nu[i, j] = 0.023*self.utility_Re[i, j]**0.8*self.utility_Pr[i, j]**0.4
        
                        if self.fuel_Re[i, j] < 2300: 
                            self.fuel_Nu[i, j] = self.entryNu(self.reactant_z[j], self.fuel_Re[i, j], self.fuel_Pr[i, j], self.fuel_f[i, j], self.reactant_eps, self.reactant_sqrtA)
                        else:
                            self.fuel_Nu[i, j] = 0.023*self.fuel_Re[i, j]**0.8*self.fuel_Pr[i, j]**0.4
                            
                        self.reactant_h[i, j] = self.reactant_Nu[i, j]*self.reactant_k[i, j]/self.reactant_sqrtA
                        self.utility_h[i, j] = self.utility_Nu[i, j]*self.utility_k[i, j]/self.utility_sqrtA
                        self.fuel_h[i, j] = self.fuel_Nu[i, j]*self.fuel_k[i, j]/self.reactant_sqrtA#calculate heat tranfer for all interactions
                        #print(self.Q_utilityPlate[i, j]/(self.metalRho*self.metalcp*self.Vcell_utilityPlate))
        
        #output utility and reactant conditions
        self.reactantout = [self.reactant_T[:, :], self.reactant_P[:, :]]
        self.utilityout = [self.utility_T[:, :], self.utility_P[:, :]]
        self.fuelout = [self.fuel_T[:, :], self.fuel_P[:, :]]
        #self.fuelout = self.T_fuelPlate
        
        return(self.reactantout, self.utilityout, self.fuelout, self.T_reactantPlate, self.T_utilityPlate, self.T_fuelPlate)   
    
    def transientHX(self, t, T):
        """
        Method to model the transient temperature response of the PCHE. 
        Requires a 1-dimensional input.

        Parameters
        ----------
        t : Float
            Time points, units of s
        T : List (floats)
            Temperature profiles for the reactant and utility channels, units of K.
            Produced by concatenating lists of temperatures produced with the 
            ravel method. 
            
            E.g. np.concatenate([reactantinitials.ravel(), utilityinitials.ravel()])

        Returns
        -------
        dTdt : List (floats)
            Differenital temperature profile, for use in an ODE solver. Units
            of K s-1.

        """
        #print(t)
        rows = int(self.dims[2])
        columns = int(self.dims[3])
        
        #unwrap temperature vector and set up dTdt 
        initial_reactant_Temps, initial_utility_Temps, initial_fuel_Temps, initial_reactantPlate_Temps, initial_utilityPlate_Temps, initial_fuelPlate_Temps = self.unwrap_T(T)
        
        dTdt_reactant = np.zeros(shape = [rows, columns])
        dTdt_utility = np.zeros(shape = [rows, columns])
        dTdt_fuel = np.zeros(shape = [rows, columns])
        
        dTdt_reactantPlate = np.zeros(shape = [rows, columns])
        dTdt_utilityPlate = np.zeros(shape = [rows, columns])
        dTdt_fuelPlate = np.zeros(shape = [rows, columns])

        #update properties and correlations
        self.update_properties(initial_reactant_Temps, initial_utility_Temps, initial_fuel_Temps)

        #create a arrays to store heat transfer between each fluid/solid
        Q_uPuF, Q_uPrF, Q_uPrP, Q_rPrF, Q_rPfP, Q_rPfF, Q_fPfF = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*7)

        #evaluate Q then dTdt for [0, 0]
        #Q_uPuF[0, 0] = self.utility_h[0, 0]*self.hx_area_uPuF*(initial_utilityPlate_Temps[0, 0] - initial_utility_Temps[0, 0])
        #Q_uPrF[0, 0] = self.reactant_h[0, 0]*self.hx_area_uPrF*(initial_reactant_Temps[0, 0] - initial_utilityPlate_Temps[0, 0])
        #Q_uPrP[0, 0] = self.metalk*self.hx_area_uPrP*(initial_reactantPlate_Temps[0, 0] - initial_utilityPlate_Temps[0, 0])
        #Q_rPrF[0, 0] = self.reactant_h[0, 0]*self.hx_area_rPrF*(initial_reactant_Temps[0, 0] - initial_reactantPlate_Temps[0, 0])
        #Q_rPfF[0, 0] = self.fuel_h[0, 0]*self.hx_area_rPfF*(initial_reactantPlate_Temps[0, 0] - initial_fuel_Temps[0, 0])
        #Q_rPfP[0, 0] = self.metalk*self.hx_area_rPfP*(initial_reactantPlate_Temps[0, 0] - initial_fuelPlate_Temps[0, 0])
        #Q_fPfF[0, 0] = self.fuel_h[0, 0]*self.hx_area_fPfF*(initial_fuel_Temps[0, 0] - initial_fuelPlate_Temps[0, 0])
        for i in range(rows):
            for j in range(columns):
                Q_uPuF[i, j] = self.utility_h[i, j]*self.hx_area_uPuF*(initial_utilityPlate_Temps[i, j] - initial_utility_Temps[i, j])
                Q_uPrF[i, j] = self.reactant_h[i, j]*self.hx_area_uPrF*(initial_reactant_Temps[i, j] - initial_utilityPlate_Temps[i, j])
                Q_uPrP[i, j] = self.metalk*self.hx_area_uPrP*(initial_reactantPlate_Temps[i, j] - initial_utilityPlate_Temps[i, j])/self.dims[5]
                Q_rPrF[i, j] = self.reactant_h[i, j]*self.hx_area_rPrF*(initial_reactant_Temps[i, j] - initial_reactantPlate_Temps[i, j])
                Q_rPfF[i, j] = self.fuel_h[i, j]*self.hx_area_rPfF*(initial_reactantPlate_Temps[i, j] - initial_fuel_Temps[i, j])
                Q_rPfP[i, j] = self.metalk*self.hx_area_rPfP*(initial_reactantPlate_Temps[i, j] - initial_fuelPlate_Temps[i, j])/self.dims[5]
                Q_fPfF[i, j] = -self.fuel_h[i, j]*self.hx_area_fPfF*(initial_fuel_Temps[i, j] - initial_fuelPlate_Temps[i, j])
                
                #if(Q_fPfF[i, j] < 0):
                #    print("bad")
        
        #evaluate [0, 0]
        dTdt_reactant[0, 0] = (self.reactant_in[1]/self.dims[2]*(self.reactant_in[2]-initial_reactant_Temps[0, 0])/(self.Vcell_reactant*self.reactant_rho[0, 0])) - (Q_uPrF[0, 0] + Q_rPrF[0, 0])/(self.reactant_cp[0,0]*self.Vcell_reactant*self.reactant_rho[0, 0])
        dTdt_utility[0, 0] = (self.utility_in[1]/self.dims[3]*(self.utility_in[2]-initial_utility_Temps[0, 0])/(self.Vcell_utility*self.utility_rho[0, 0])) + (Q_uPuF[0, 0])/(self.utility_cp[0,0]*self.Vcell_utility*self.utility_rho[0, 0])
        dTdt_fuel[0, 0] = (self.fuel_in[1]/self.dims[2]*(self.fuel_in[2]-initial_fuel_Temps[0, 0])/(self.Vcell_fuel*self.fuel_rho[0, 0])) + (Q_rPfF[0, 0] + Q_fPfF[0, 0])/(self.fuel_cp[0,0]*self.Vcell_fuel*self.fuel_rho[0, 0])
        dTdt_reactantPlate[0, 0] = (Q_rPrF[0, 0] - Q_rPfF[0, 0] - Q_rPfP[0, 0] - Q_uPrP[0, 0])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
        dTdt_utilityPlate[0, 0] = (Q_uPrF[0, 0] + Q_uPrP[0, 0] - Q_uPuF[0, 0])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
        dTdt_fuelPlate[0, 0] = (Q_rPfP[0, 0] - Q_fPfF[0, 0])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)

        #set initial column (steam channel 1)
        for i in range(1, rows):
            dTdt_reactant[i, 0] = (self.reactant_in[1]/self.dims[2]*(self.reactant_in[2]-initial_reactant_Temps[i, 0])/(self.Vcell_reactant*self.reactant_rho[i, 0])) - (Q_uPrF[i, 0] + Q_rPrF[i, 0])/(self.reactant_cp[i,0]*self.Vcell_reactant*self.reactant_rho[i, 0])
            dTdt_utility[i, 0] = (self.utility_in[1]/self.dims[3]*(initial_utility_Temps[i-1, 0]-initial_utility_Temps[i, 0])/(self.Vcell_utility*self.utility_rho[i, 0])) + (Q_uPuF[i, 0])/(self.utility_cp[i,0]*self.Vcell_utility*self.utility_rho[i, 0])
            dTdt_fuel[i, 0] = (self.fuel_in[1]/self.dims[2]*(self.fuel_in[2]-initial_fuel_Temps[i, 0])/(self.Vcell_fuel*self.fuel_rho[i, 0])) + (Q_rPfF[i, 0] + Q_fPfF[i, 0])/(self.fuel_cp[i,0]*self.Vcell_fuel*self.fuel_rho[i, 0])
            dTdt_reactantPlate[i, 0] = (Q_rPrF[i, 0] - Q_rPfF[i, 0] - Q_rPfP[i, 0] - Q_uPrP[i, 0])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
            dTdt_utilityPlate[i, 0] = (Q_uPrF[i, 0] + Q_uPrP[i, 0] - Q_uPuF[i, 0])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
            dTdt_fuelPlate[i, 0] = (Q_rPfP[i, 0] - Q_fPfF[i, 0])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)    

        #set initial row (reactant channel 1)
        for i in range(1, columns):
            dTdt_reactant[0, i] = (self.reactant_in[1]/self.dims[2]*(initial_reactant_Temps[0, i-1]-initial_reactant_Temps[0, i])/(self.Vcell_reactant*self.reactant_rho[0, i])) - (Q_uPrF[0, i] + Q_rPrF[0, i])/(self.reactant_cp[0,i]*self.Vcell_reactant*self.reactant_rho[0, i])
            dTdt_utility[0, i] = (self.utility_in[1]/self.dims[3]*(self.utility_in[2]-initial_utility_Temps[0, i])/(self.Vcell_utility*self.utility_rho[0, i])) + (Q_uPuF[0, i])/(self.utility_cp[0,i]*self.Vcell_utility*self.utility_rho[0, i])
            dTdt_fuel[0, i] = (self.fuel_in[1]/self.dims[2]*(initial_fuel_Temps[0, i-1]-initial_fuel_Temps[0, i])/(self.Vcell_fuel*self.fuel_rho[0, i])) + (Q_rPfF[0, i] + Q_fPfF[0, i])/(self.fuel_cp[0,i]*self.Vcell_fuel*self.fuel_rho[0, i])
            dTdt_reactantPlate[0, i] = (Q_rPrF[0, i] - Q_rPfF[0, i] - Q_rPfP[0, i] - Q_uPrP[0, i])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
            dTdt_utilityPlate[0, i] = (Q_uPrF[0, i] + Q_uPrP[0, i] - Q_uPuF[0, i])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
            dTdt_fuelPlate[0, i] = (Q_rPfP[0, i] - Q_fPfF[0, i])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
    
        #solve remainder of arrays
        for j in range(1, columns):
            for i in range(1, rows):
                dTdt_reactant[i, j] = (self.reactant_in[1]/self.dims[2]*(initial_reactant_Temps[i, j-1]-initial_reactant_Temps[i, j])/(self.Vcell_reactant*self.reactant_rho[i, j])) + (-Q_uPrF[i, j] - Q_rPrF[i, j])/(self.reactant_cp[i,j]*self.Vcell_reactant*self.reactant_rho[i, j])
                dTdt_utility[i, j] = (self.utility_in[1]/self.dims[3]*(initial_utility_Temps[i-1, j]-initial_utility_Temps[i, j])/(self.Vcell_utility*self.utility_rho[i, j])) + (Q_uPuF[i, j])/(self.utility_cp[i,j]*self.Vcell_utility*self.utility_rho[i, j])
                dTdt_fuel[i, j] = (self.fuel_in[1]/self.dims[2]*(initial_fuel_Temps[i, j-1]-initial_fuel_Temps[i, j])/(self.Vcell_fuel*self.fuel_rho[i, j])) + (Q_rPfF[i, j] + Q_fPfF[i, j])/(self.fuel_cp[i,j]*self.Vcell_fuel*self.fuel_rho[i, j])
                dTdt_reactantPlate[i, j] = (Q_rPrF[i, j] - Q_rPfF[i, j] - Q_rPfP[i, j] - Q_uPrP[i, j])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
                dTdt_utilityPlate[i, j] = (Q_uPrF[i, j] + Q_uPrP[i, j] - Q_uPuF[i, j])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
                dTdt_fuelPlate[i, j] = (Q_rPfP[i, j] - Q_fPfF[i, j])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
                
                
                
        #intra-plate condictive heat transfer - set up body of the heat exchanger
        for j in range(1, columns-1):
            for i in range(1, rows-1):
                dTdt_reactantPlate[i, j] = dTdt_reactantPlate[i, j] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_reactantPlate_Temps[i-1, j] - initial_reactantPlate_Temps[i+1, j])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
                dTdt_reactantPlate[i, j] = dTdt_reactantPlate[i, j] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_reactantPlate_Temps[i, j-1] - initial_reactantPlate_Temps[i, j+1])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
                dTdt_fuelPlate[i, j] = dTdt_fuelPlate[i, j] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_fuelPlate_Temps[i-1, j] - initial_fuelPlate_Temps[i+1, j])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
                dTdt_fuelPlate[i, j] = dTdt_fuelPlate[i, j] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_fuelPlate_Temps[i, j-1] - initial_fuelPlate_Temps[i, j+1])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
                dTdt_utilityPlate[i, j] = dTdt_utilityPlate[i, j] + ((self.dims[1]+self.dims[4])*self.dims[5]-self.utility_sqrtA**2)*self.metalk/(self.dims[0]+self.dims[4])*(initial_utilityPlate_Temps[i-1, j] - initial_utilityPlate_Temps[i+1, j])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
                dTdt_utilityPlate[i, j] = dTdt_utilityPlate[i, j] + ((self.dims[0]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[1]+self.dims[4])*(initial_utilityPlate_Temps[i, j-1] - initial_utilityPlate_Temps[i, j+1])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
        
        # #add insulated boundaries

        #fix my bad coding
        rows = rows-1
        columns = columns-1
        
        for i in range(1, columns-1):
            dTdt_reactantPlate[0, i] = dTdt_reactantPlate[0, i] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_reactantPlate_Temps[0, i-1] - initial_reactantPlate_Temps[0, i+1])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
            dTdt_reactantPlate[rows, i] = dTdt_reactantPlate[rows, i] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_reactantPlate_Temps[rows, i-1] - initial_reactantPlate_Temps[rows, i+1])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
            dTdt_reactantPlate[0, i] = dTdt_reactantPlate[0, i] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_reactantPlate_Temps[0, i] - initial_reactantPlate_Temps[1, i])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
            dTdt_reactantPlate[rows, i] = dTdt_reactantPlate[rows, i] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_reactantPlate_Temps[rows-1, i] - initial_reactantPlate_Temps[rows, i])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)

            dTdt_fuelPlate[0, i] = dTdt_fuelPlate[0, i] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_fuelPlate_Temps[0, i-1] - initial_fuelPlate_Temps[0, i+1])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
            dTdt_fuelPlate[rows, i] = dTdt_fuelPlate[rows, i] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_fuelPlate_Temps[rows, i-1] - initial_fuelPlate_Temps[rows, i+1])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
            dTdt_fuelPlate[0, i] = dTdt_fuelPlate[0, i] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_fuelPlate_Temps[0, i] - initial_fuelPlate_Temps[1, i])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
            dTdt_fuelPlate[rows, i] = dTdt_fuelPlate[rows, i] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_fuelPlate_Temps[rows-1, i] - initial_fuelPlate_Temps[rows, i])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)

            dTdt_utilityPlate[0, i] = dTdt_utilityPlate[0, i] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_utilityPlate_Temps[0, i-1] - initial_utilityPlate_Temps[0, i+1])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
            dTdt_utilityPlate[rows, i] = dTdt_utilityPlate[rows, i] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_utilityPlate_Temps[rows, i-1] - initial_utilityPlate_Temps[rows, i+1])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
            dTdt_utilityPlate[0, i] = dTdt_utilityPlate[0, i] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.utility_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_utilityPlate_Temps[0, i] - initial_utilityPlate_Temps[1, i])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
            dTdt_utilityPlate[rows, i] = dTdt_utilityPlate[rows, i] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.utility_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_utilityPlate_Temps[rows-1, i] - initial_utilityPlate_Temps[rows, i])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)

        for j in range(1, rows-1):
            dTdt_reactantPlate[j, 0] = dTdt_reactantPlate[j, 0] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_reactantPlate_Temps[j-1, 0] - initial_reactantPlate_Temps[j+1, 0])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
            dTdt_reactantPlate[j, columns] = dTdt_reactantPlate[j, columns] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_reactantPlate_Temps[j-1, columns] - initial_reactantPlate_Temps[j+1, columns])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
            dTdt_reactantPlate[j, 0] = dTdt_reactantPlate[j, 0] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_reactantPlate_Temps[j, 0] - initial_reactantPlate_Temps[j, 1])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)
            dTdt_reactantPlate[j, columns] = dTdt_reactantPlate[j, columns] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_reactantPlate_Temps[j, columns-1] - initial_reactantPlate_Temps[j, columns])/(self.metalcp*self.metalRho*self.Vcell_reactantPlate)

            dTdt_fuelPlate[j, 0] = dTdt_fuelPlate[j, 0] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_fuelPlate_Temps[j-1, 0] - initial_fuelPlate_Temps[j+1, 0])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
            dTdt_fuelPlate[j, columns] = dTdt_fuelPlate[j, columns] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.reactant_sqrtA**2)*self.metalk/(self.dims[1]+self.dims[4])*(initial_fuelPlate_Temps[j-1, columns] - initial_fuelPlate_Temps[j+1, columns])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
            dTdt_fuelPlate[j, 0] = dTdt_fuelPlate[j, 0] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_fuelPlate_Temps[j, 0] - initial_fuelPlate_Temps[j, 1])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)
            dTdt_fuelPlate[j, columns] = dTdt_fuelPlate[j, columns] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_fuelPlate_Temps[j, columns-1] - initial_fuelPlate_Temps[j, columns])/(self.metalcp*self.metalRho*self.Vcell_fuelplate)

            dTdt_utilityPlate[j, 0] = dTdt_utilityPlate[j, 0] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_utilityPlate_Temps[j-1, 0] - initial_utilityPlate_Temps[j+1, 0])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
            dTdt_utilityPlate[j, columns] = dTdt_utilityPlate[j, columns] + ((self.dims[1]+self.dims[4])*self.dims[5])*self.metalk/(self.dims[0]+self.dims[4])*(initial_utilityPlate_Temps[j-1, columns] - initial_utilityPlate_Temps[j+1, columns])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
            dTdt_utilityPlate[j, 0] = dTdt_utilityPlate[j, 0] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.utility_sqrtA**2)*self.metalk/(self.dims[0]+self.dims[4])*(initial_utilityPlate_Temps[j, 0] - initial_utilityPlate_Temps[j, 1])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)
            dTdt_utilityPlate[j, columns] = dTdt_utilityPlate[j, columns] + ((self.dims[0]+self.dims[4])*self.dims[5]-self.utility_sqrtA**2)*self.metalk/(self.dims[0]+self.dims[4])*(initial_utilityPlate_Temps[j, columns-1] - initial_utilityPlate_Temps[j, columns])/(self.metalcp*self.metalRho*self.Vcell_utilityPlate)

        #wrap arrays up as a vector and return                
        dTdt = np.concatenate([dTdt_reactant.ravel(), dTdt_utility.ravel(), dTdt_fuel.ravel(), dTdt_reactantPlate.ravel(), dTdt_utilityPlate.ravel(), dTdt_fuelPlate.ravel()])
        return dTdt

#set input dimensions and streams    
reactantin = [{'O2':5, 'CO2':92, 'H2O':3, 'CH4':0}, 1/116/6, 873, 1500000]
utilityin = [{'H2O': 100}, 1/116/6, 400, 1000000]
fuelin = [{'CH4': 100}, 1/58/250/6, 400, 1500000]
dimensions = [0.0015, 0.0015, 30, 20, 0.0011, 0.0021]

#solve steady state model
hx1 = crossflow_hx(reactantin, utilityin, fuelin, dimensions)
reactantout, utilityout, fuelout, reactantplate, utilityplate, fuelplate = hx1.solvehx()

#solve transient model - grab initial feed temperatures and use as input temperature
reactantinitials = reactantin[2]*np.ones(shape = (dimensions[2], dimensions[3]))
utilityinitials = utilityin[2]*np.ones(shape = (dimensions[2], dimensions[3]))
fuelinitials = fuelin[2]*np.ones(shape = (dimensions[2], dimensions[3]))

reactantplate = reactantplate[:, 1:]
utilityplate = utilityplate[1:, :]
fuelplate = fuelplate[:, 1:]

#use this to take SS model output as initial conditions
#initialTemps = np.concatenate([reactantout[0].ravel(), utilityout[0].ravel()])

#use this to take initial input conditions as initial conditions
initialTemps = np.concatenate([reactantinitials.ravel(), utilityinitials.ravel(), fuelinitials.ravel(), reactantplate.ravel(), utilityplate.ravel(), fuelplate.ravel()])

#solve using an implicit solver (BDF)
sol = solve_ivp(hx1.transientHX, [0, 0.05], initialTemps, method = "RK23", t_eval = [0, 0.001, 0.005, 0.01, 0.025, 0.05])

#grab final temperature profiles and reshape into given form
#T_averages_reactants = np.zeros((1, dimensions[3]))
#_averages_utility = np.zeros((1, dimensions[2]))

# for i in range(6):
#     n = i #take column 2
#     reactant_T_out = sol['y'][0:(dimensions[2]*(dimensions[3]))]
#     reactant_T_out = reactant_T_out[:, n]
#     reactant_T_out = reactant_T_out.reshape(dimensions[2], dimensions[3])
    
#     utility_T_out = sol['y'][(dimensions[2]*(dimensions[3])):dimensions[2]*(dimensions[3])*2]
#     utility_T_out = utility_T_out[:, n]
#     utility_T_out = utility_T_out.reshape(dimensions[2], dimensions[3])
    
#     fuel_T_out = sol['y'][(dimensions[2]*(dimensions[3]))*2:dimensions[2]*(dimensions[3])*3]
#     fuel_T_out = fuel_T_out[:, n]
#     fuel_T_out = fuel_T_out.reshape(dimensions[2], dimensions[3])
    
#     reactantPlate_T_out = sol['y'][(dimensions[2]*(dimensions[3]))*3:dimensions[2]*(dimensions[3])*4]
#     reactantPlate_T_out = reactantPlate_T_out[:, n]
#     reactantPlate_T_out = reactantPlate_T_out.reshape(dimensions[2], dimensions[3])
    
#     utilityPlate_T_out = sol['y'][(dimensions[2]*(dimensions[3]))*4:dimensions[2]*(dimensions[3])*5]
#     utilityPlate_T_out = utilityPlate_T_out[:, n]
#     utilityPlate_T_out = utilityPlate_T_out.reshape(dimensions[2], dimensions[3])
    
#     fuelPlate_T_out = sol['y'][(dimensions[2]*(dimensions[3]))*5:dimensions[2]*(dimensions[3])*6]
#     fuelPlate_T_out = fuelPlate_T_out[:, n]
#     fuelPlate_T_out = fuelPlate_T_out.reshape(dimensions[2], dimensions[3])
    
#     reactant_T_out_avg = reactant_T_out.mean(axis = 0)
#     utility_T_out_avg = utility_T_out.mean(axis = 1).transpose()
#     fuel_T_out_avg = fuel_T_out.mean(axis = 0)
#     reactantPlate_T_out_avg = reactantPlate_T_out.mean(axis = 0)
#     utilityPlate_T_out_avg = utilityPlate_T_out.mean(axis = 1).transpose()
#     fuelPlate_T_out_avg = fuelPlate_T_out.mean(axis = 0)
    
#     T_averages_reactants = np.vstack((T_averages_reactants, reactant_T_out_avg, fuel_T_out_avg, reactantPlate_T_out_avg, fuelPlate_T_out_avg))
#     T_averages_utility = np.vstack((T_averages_utility, utility_T_out_avg, utilityPlate_T_out_avg))
    
#     np.savetxt('T_profile_reactants.csv', T_averages_reactants, delimiter=',')
#     np.savetxt('T_profile_utility.csv', T_averages_utility, delimiter=',')
#     #for full profile export
#     #Tprofile = np.ones((dimensions[2], dimensions[3]))
#     #Tprofile = np.concatenate([Tprofile, reactant_T_out, utility_T_out, fuel_T_out, reactantPlate_T_out, utilityPlate_T_out, fuelPlate_T_out])
#     #np.savetxt('T_profile_t' + str(sol['t'][i]), Tprofile, delimiter=',')
# np.savetxt('T_profile_reactants.csv', T_averages_reactants, delimiter=',')
# np.savetxt('T_profile_utility.csv', T_averages_utility, delimiter=',')
    
    

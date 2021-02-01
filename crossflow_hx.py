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
    fuel_in : list of four elements
        1. fuel composition, Cantera library of compositions, units of mol%
        2. mass flow rate through fuel plate, units of kg/s 
        3. fuel inlet temperature, units K        
        4. fuel inlet absolute pressure, units of Pa
    
    utility_in : list of four elements
        1. utility composition, Cantera library of compositions, units of mol%        
        2. mass flow rate through utility plate, units of kg/s         
        3. utility inlet temperature, units K        
        4. utility inlet absolute pressure, units of Pa
        
    dims : list of five elements
        1. fuel channel diameter, units of m
        2. utility channel diameter [m], units of m
        3. number of fuel channels, dimensionless
        4. number of utility channels, dimensionless
        5. wall thickness between channels, units of m
        
        
    Returns
    -----
    solve_hx method : Returns two arrays
        array1 : [fuel temperature profile, fuel pressure profile]
            fuel temperature profile: cell-by-cell array with inlet shown in 
            column 0, units of K
            
            fuel pressure profile: cell-by-cell array with inlet shown in 
            column 0, units of Pa
        array2 : [utility temperature profile, utility pressure profile]
            fuel temperature profile: cell-by-cell array with inlet shown in 
            row 0, units of K
            
            fuel pressure profile: cell-by-cell array with inlet shown in 
            row 0, units of Pa
            
    transientHX method : For use with an implicit ODE solver
    
    Reference
    -----
    See other methods in class.
    
    Applicability
    -----
    Applicable for laminar flow (Re < 2300).
    
    Not suitable for use beyond approximately ideal conditions.
            
    """
    
    def __init__(self, fuel_in, utility_in, dims):
        #list of [[stream composition], mass flow [kg/s], Temp [K], Pressure [Pa abs]]
        self.fuel_in = fuel_in
        self.utility_in = utility_in
        #dimensions are [fuel channel diameter, utility channel diameter, number fuel channels, number utility channels, wall thicknesses for channels]
        self.dims = dims


        #cross-sectional area, hydraulic diameter, square root of cross section
        self.fuel_cs = math.pi*self.dims[0]**2/8
        self.utility_cs = math.pi*self.dims[1]**2/8
        self.fuel_dh = 4*self.fuel_cs/(math.pi*self.dims[0]/2 + self.dims[0])
        self.utility_dh = 4*self.utility_cs/(math.pi*self.dims[1]/2 + self.dims[1])
        self.fuel_sqrtA = self.fuel_cs**0.5
        self.utility_sqrtA = self.utility_cs**0.5
        
        self.fuel_eps = 0.5 #semicircular channel aspect ratio
        self.utility_eps = 0.5
        
        #revise this for more complex modelling?
        self.hx_area = self.dims[0]*self.dims[1]
        
        #set up initial cantera fluids - gri30 good for combustion of natural gas
        #not setting up initial fluid conditions here
        self.fuel = ct.Solution('gri30.xml')
        self.utility = ct.Solution('gri30.xml')
        self.fuel.transport_model = 'Mix'
        self.utility.transport_model = 'Mix'
        
        """
        set up inital arrays to reserve memory
        needed arrays are temperature, pressure, density, velocity, viscosity, 
        reynolds number, specific heat capacity, thermal conductivity, 
        prandtl number, nusselt number, convective heat transfer coefficient, 
        overall heat transfer coefficient, total heat transfer, z for hx calcs length
        """
        
        #create empty identical arrays for each variable - different dimensions for each fluid
        self.fuel_T, self.fuel_P, self.fuel_rho, self.fuel_u, self.fuel_mu, self.fuel_Re, self.fuel_cp, self.fuel_k, self.fuel_Pr, self.fuel_Nu, self.fuel_h = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*11)
        self.utility_T, self.utility_P, self.utility_rho, self.utility_u, self.utility_mu, self.utility_Re, self.utility_cp, self.utility_k, self.utility_Pr, self.utility_Nu, self.utility_h = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 2)]*11)
        self.U, self.Q = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*2)
        
        #axial position
        self.fuel_z = np.linspace(start = 0, stop = self.dims[3]*(self.dims[1] + self.dims[4]) - self.dims[4], num = self.dims[3] + 1)
        self.utility_z = np.linspace(start = 0, stop = self.dims[2]*(self.dims[0] + self.dims[4]) - self.dims[4], num = self.dims[2] + 1)
        
        #setup for pressure drop correlations
        self.fuel_lplus, self.fuel_f = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 1)]*2)
        self.utility_lplus, self.utility_f = map(np.copy, [self.emptyarray(self.dims[2], self.dims[3], 2)]*2)
        
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
        #extra column is for fuel channel initialization, and extra row is for utility channel
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
    
    def fluid_properties_fuel(self, T, P):
        """
        Use Cantera to evaluate fluid properties for given conditions. 
        Uses the fuel gas object to reduce composition updates.

        Parameters
        ----------
        T : Float
            Fuel temperature, units of K.
        P : Float
            Fuel pressure, units of Pa.

        Returns
        -------
        rho : Float
            Fuel density, units of kg m-3
        mu : Float
            Fuel viscosity, units of kg m-1 s-1
        cp : Float
            Fuel specific heat capacity, mass basis, units of J kg-1 K-1
        k : Float
            Fuel thermal conducivity, units of W m-1 K-1
        Pr : Float
            Fuel Prandtl number, dimensionless
            
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
    
    def update_properties(self, T_fuel, T_utility):
        """
        Update arrays of properties and convective heat transfer correlations 
        for new temperature profiles. For use in the transient solver. 
        Assuming pressures are constant from the steady-state model.

        Parameters
        ----------
        T_fuel : Array
            Fuel temperatures, float, units of K.
        T_utility : Array
            Utility temperatures, float, units of K.

        Returns
        -------
        None.

        """
        for j in range(self.dims[3]):
            for i in range(self.dims[2]):
                self.fuel_rho[i, j], self.fuel_mu[i, j], self.fuel_cp[i, j],self.fuel_k[i, j], self.fuel_Pr[i, j] = self.fluid_properties_fuel(T_fuel[i, j], self.fuel_P[i, j])
                self.utility_rho[i, j], self.utility_mu[i, j], self.utility_cp[i, j],self.utility_k[i, j], self.utility_Pr[i, j] = self.fluid_properties_utility(T_utility[i, j], self.utility_P[i, j])
                
                self.fuel_u[i, j] = self.fuel_u0*self.fuel_rho0/self.fuel_rho[i, j]
                self.utility_u[i, j] = self.utility_u0*self.utility_rho0/self.utility_rho[i, j]
                
                self.fuel_Re[i, j] = self.reynolds(self.fuel_rho[i, j], self.fuel_u[i, j], self.fuel_sqrtA, self.fuel_mu[i, j])
                self.utility_Re[i, j] = self.reynolds(self.utility_rho[i, j], self.utility_u[i, j], self.utility_sqrtA, self.utility_mu[i, j])
                
                if self.fuel_Re[i, j] < 2300: 
                    self.fuel_Nu[i, j] = self.entryNu(self.fuel_z[j], self.fuel_Re[i, j], self.fuel_Pr[i, j], self.fuel_f[i, j], self.fuel_eps, self.fuel_sqrtA)
                else:
                    self.fuel_Nu[i, j] = 0.023*self.fuel_Re[i, j]**0.8*self.fuel_Pr[i, j]**0.3
                    
                if self.utility_Re[i, j] < 2300:
                    self.utility_Nu[i, j] = self.entryNu(self.utility_z[i], self.utility_Re[i, j], self.utility_Pr[i, j], self.utility_f[i, j], self.utility_eps, self.utility_sqrtA)
                else:
                    self.utility_Nu[i, j] = 0.023*self.utility_Re[i, j]**0.8*self.utility_Pr[i, j]**0.4
                
                self.fuel_h[i, j] = self.fuel_Nu[i, j]*self.fuel_k[i, j]/self.fuel_sqrtA
                self.utility_h[i, j] = self.utility_Nu[i, j]*self.utility_k[i, j]/self.utility_sqrtA
                
                self.U[i, j] = 1/(1/self.fuel_h[i, j] + 1/self.utility_h[i, j] + 0.0005/45)  
                
    def unwrap_T(self, T_vector):
        """
        Used to manipulate vector of fuel and utility temperatures profiles
        into two arrays. Manipulates data into easily iterable form after 
        passing through the 1-dimensional ODE solver.

        Parameters
        ----------
        T_vector : List
            Temperature profile produced by ODE solver, units of K.

        Returns
        -------
        initial_fuel_Temps : Array
            2-Dimensional fuel plate temperature profile, with column 0 as the 
            inlet. Units of K.
        initial_utility_Temps : Array
            2-Dimensional utility plate temperature profile, with row 0 as the 
            inlet. Units of K.

        """
        initial_fuel_Temps = T_vector[0:int(self.dims[2]*self.dims[3])].reshape(int(self.dims[2]), int(self.dims[3]))
        initial_utility_Temps = T_vector[int(self.dims[2]*self.dims[3]):].reshape(int(self.dims[2]), int(self.dims[3]))
        
        return initial_fuel_Temps, initial_utility_Temps
        
    def solvehx(self):
        """
        Method to solve steady-state temperature and pressure profiles for fuel
        and utility channels. Only solves when called. 

        Returns
        -------
        fuelout : Array
            element1: Temperature profile for fuel plate, with inlet in column 
            0 and outlet in column n. Units of K.
            
            element2: Pressure profile for fuel plate, with inlet in column 0 
            and outlet in column n. Units of Pa abs.
        
        utilityout : Array
            element1: Temperature profile for utility plate, with inlet in 
            row 0 and outlet in row n. Units of K.
            
            element2: Pressure profile for utility plate, with inlet in row 0 
            and outlet in row n. Units of Pa abs.
                
        """
        #set initial fluid conditions and arrays
        self.fuel.TPY = self.fuel_in[2], self.fuel_in[3], self.fuel_in[0]
        self.utility.TPY = self.utility_in[2], self.utility_in[3], self.utility_in[0]
        
        #ignore friction for now on pressure. assume no gas-phase reactions
        self.fuel_T[:, 0] = self.fuel_in[2]
        self.fuel_P[:, 0] = self.fuel_in[3]
        self.utility_T[0, :] = self.utility_in[2]
        self.utility_P[0, :] = self.utility_in[3]
        
        #use cantera to determine fluid properties
        self.fuel_rho[:, 0], self.fuel_mu[:, 0], self.fuel_cp[:, 0], self.fuel_k[:, 0], self.fuel_Pr[:, 0] = self.fluid_properties_fuel(self.fuel_in[2], self.fuel_in[3])
        self.utility_rho[0, :], self.utility_mu[0, :], self.utility_cp[0, :], self.utility_k[0, :], self.utility_Pr[0, :] = self.fluid_properties_utility(self.utility_in[2], self.utility_in[3])
        
        #inlet velocity = mass flow rate in / density / # channels / channel cross section
        self.fuel_u[:, 0] = self.fuel_in[1]/(self.fuel_rho[0, 0]*self.dims[2]*self.fuel_cs)
        self.utility_u[0, :] = self.utility_in[1]/(self.utility.density*self.dims[3]*self.utility_cs)

        #grab initial velocity and density to determine later velocity via continuity equation
        self.fuel_u0 = self.fuel_u[0, 0]    
        self.utility_u0 = self.utility_u[0, 0]
        self.fuel_rho0 = self.fuel_rho[0, 0]
        self.utility_rho0 = self.utility_rho[0, 0]

        #maybe update this soon?
        self.fuel_Re[:, 0] = self.reynolds(self.fuel_rho[0, 0], self.fuel_u[0, 0], self.fuel_sqrtA, self.fuel_mu[0, 0])
        self.utility_Re[0, :] = self.reynolds(self.utility_rho[0, 0], self.utility_u[0, 0], self.utility_sqrtA, self.utility_mu[0, 0])
        
        #Muzychka & Yovanovich - entry region friction factor
        #L+ is 0 for the entry region
        self.fuel_f[:, 0] = self.devflow_friction(self.fuel_lplus[0, 0], self.fuel_eps, self.fuel_Re[0, 0])
        self.utility_f[0, :] = self.devflow_friction(self.utility_lplus[0, 0], self.utility_eps, self.utility_Re[0, 0])
        
        #Muzychka & Yovanovich - combined entry region for small z*
        self.fuel_Nu[:, 0] = self.entryNu(0, self.fuel_Re[0, 0], self.fuel_Pr[0, 0], self.fuel_f[0, 0], self.fuel_eps, self.fuel_sqrtA)
        self.utility_Nu[0, :] = self.entryNu(0, self.utility_Re[0, 0], self.utility_Pr[0, 0], self.utility_f[0, 0], self.utility_eps, self.utility_sqrtA)
        
        #should this be sqrtA or dh?
        self.fuel_h[:, 0] = self.fuel_Nu[0, 0] * self.fuel_k[0, 0]/self.fuel_sqrtA
        self.utility_h[0, :] = self.utility_Nu[0, 0] * self.utility_k[0, 0]/self.utility_sqrtA
        
        #now iterate through and solve for everything
        for j in range(0, self.dims[3]):
            for i in range(0, self.dims[2]):
                #calculate heat tranfer coefficient and total heat transfer
                self.U[i, j] = 1/(1/self.fuel_h[i, j] + 1/self.utility_h[i, j] + 0.0005/45)
                self.Q[i, j] = self.U[i, j]*self.hx_area*(self.fuel_T[i, j] - self.utility_T[i, j])
                
                #use the mass flow rate through each channel instead of the velocity in each channel for the heat transferred
                self.fuel_T[i, j+1] = self.fuel_T[i, j] - self.Q[i, j]/(self.fuel_in[1]/self.dims[2]*self.fuel_cp[i, j])
                self.utility_T[i+1, j] = self.utility_T[i, j] + self.Q[i, j]/(self.utility_in[1]/self.dims[3]*self.utility_cp[i, j])
                
                #assuming incompressible flow given low velocity
                self.fuel_P[i, j+1] = self.fuel_P[i, j] - 2*self.fuel_f[i, j] * (self.dims[1] + self.dims[4]) / self.fuel_dh *self.fuel_rho[i, j]*self.fuel_u[i, j]**2
                self.utility_P[i+1, j] = self.utility_P[i, j] - 2*self.utility_f[i, j] * (self.dims[0] + self.dims[4]) / self.utility_dh *self.utility_rho[i, j]*self.utility_u[i, j]**2
        
                #update thermophysical and transport properties for new conditions
                self.fuel_rho[i, j+1], self.fuel_mu[i, j+1], self.fuel_cp[i, j+1],self.fuel_k[i, j+1], self.fuel_Pr[i, j+1] = self.fluid_properties_fuel(self.fuel_T[i, j+1], self.fuel_P[i, j+1])
                self.utility_rho[i+1, j], self.utility_mu[i+1, j], self.utility_cp[i+1, j],self.utility_k[i+1, j], self.utility_Pr[i+1, j] = self.fluid_properties_utility(self.utility_T[i+1, j], self.utility_P[i+1, j])
                
                #update velocity
                self.fuel_u[i, j+1] = self.fuel_u[i, j]*self.fuel_rho[i, j]/self.fuel_rho[i, j+1]
                self.utility_u[i+1, j] = self.utility_u[i, j]*self.utility_rho[i, j]/self.utility_rho[i+1, j]
                
                self.fuel_Re[i, j+1] = self.reynolds(self.fuel_rho[i, j+1], self.fuel_u[i, j+1], self.fuel_sqrtA, self.fuel_mu[i, j+1])
                self.utility_Re[i+1, j] = self.reynolds(self.utility_rho[i+1, j], self.utility_u[i+1, j], self.utility_sqrtA, self.utility_mu[i+1, j])
                
                #update lplus and friction factors
                self.fuel_lplus[i, j+1] = self.lplus(self.fuel_z[j+1], self.fuel_Re[i, j+1], self.fuel_sqrtA)
                self.utility_lplus[i+1, j] = self.lplus(self.utility_z[i+1], self.utility_Re[i+1, j], self.utility_sqrtA)
                self.fuel_f[i, j+1] = self.devflow_friction(self.fuel_lplus[i, j+1], self.fuel_eps, self.fuel_Re[i, j+1])
                self.utility_f[i+1, j] = self.devflow_friction(self.utility_lplus[i+1, j], self.utility_eps, self.utility_Re[i+1, j])
                
                #determine developing region heat transfer coefficients
                if self.fuel_Re[i, j+1] < 2300: 
                    self.fuel_Nu[i, j+1] = self.entryNu(self.fuel_z[j+1], self.fuel_Re[i, j+1], self.fuel_Pr[i, j+1], self.fuel_f[i, j+1], self.fuel_eps, self.fuel_sqrtA)
                else:
                    self.fuel_Nu[i, j+1] = 0.023*self.fuel_Re[i, j+1]**0.8*self.fuel_Pr[i, j+1]**0.3
                    
                if self.utility_Re[i+1, j] < 2300:
                    self.utility_Nu[i+1, j] = self.entryNu(self.utility_z[i+1], self.utility_Re[i+1, j], self.utility_Pr[i+1, j], self.utility_f[i+1, j], self.utility_eps, self.utility_sqrtA)
                else:
                    self.utility_Nu[i+1, j] = 0.023*self.utility_Re[i+1, j]**0.8*self.utility_Pr[i+1, j]**0.4
                
                self.fuel_h[i, j+1] = self.fuel_Nu[i, j+1]*self.fuel_k[i, j+1]/self.fuel_sqrtA
                self.utility_h[i+1, j] = self.utility_Nu[i+1, j]*self.utility_k[i+1, j]/self.utility_sqrtA
                
        #output utility and fuel conditions
        self.fuelout = [self.fuel_T[:, :], self.fuel_P[:, :]]
        self.utilityout = [self.utility_T[:, :], self.utility_P[:, :]]
        return(self.fuelout, self.utilityout)   
    
    def transientHX(self, t, T):
        """
        Method to model the transient temperature response of the PCHE. 
        Requires a 1-dimensional input.

        Parameters
        ----------
        t : Float
            Time points, units of s
        T : List (floats)
            Temperature profiles for the fuel and utility channels, units of K.
            Produced by concatenating lists of temperatures produced with the 
            ravel method. 
            
            E.g. np.concatenate([fuelinitials.ravel(), utilityinitials.ravel()])

        Returns
        -------
        dTdt : List (floats)
            Differenital temperature profile, for use in an ODE solver. Units
            of K s-1.

        """
        rows = int(self.dims[2])
        columns = int(self.dims[3])
        
        #unwrap temperature vector and set up dTdt 
        initial_fuel_Temps, initial_utility_Temps = self.unwrap_T(T)
        dTdt_fuel = np.zeros(shape = [rows, columns])
        dTdt_utility = np.zeros(shape = [rows, columns])

        #update properties and correlations
        self.update_properties(initial_fuel_Temps, initial_utility_Temps)

        #set [0, 0]
        dTdt_fuel[0, 0] = (self.fuel_in[1]/self.dims[2]*(self.fuel_in[2] - initial_fuel_Temps[0, 0])/(self.fuel_cs*(self.dims[1])*self.fuel_rho[0, 0]) - self.U[0, 0]*self.hx_area*(initial_fuel_Temps[0, 0] - initial_utility_Temps[0, 0])/(self.fuel_cp[0,0]*(self.fuel_cs*(self.dims[1])*self.fuel_rho[0, 0])))
        dTdt_utility[0, 0] =  (self.utility_in[1]/self.dims[3]*(self.utility_in[2] - initial_utility_Temps[0, 0])/(self.utility_cs*(self.dims[0])*self.utility_rho[0, 0]) + self.U[0, 0]*self.hx_area*(initial_fuel_Temps[0, 0] - initial_utility_Temps[0, 0])/(self.utility_cp[0, 0]*(self.utility_cs*(self.dims[0])*self.utility_rho[0, 0])))
                
        #set initial column (steam channel 1)
        for i in range(1, rows):
            dTdt_fuel[i, 0] = (self.fuel_in[1]/self.dims[2]*(self.fuel_in[2] - initial_fuel_Temps[i, 0])/(self.fuel_cs*(self.dims[1])*self.fuel_rho[i, 0]) - self.U[i, 0]*self.hx_area*(initial_fuel_Temps[i, 0] - initial_utility_Temps[i, 0])/(self.fuel_cp[i,0]*(self.fuel_cs*(self.dims[1])*self.fuel_rho[i, 0])))
            dTdt_utility[i, 0] =  (self.utility_in[1]/self.dims[3]*(initial_utility_Temps[i-1, 0] - initial_utility_Temps[i, 0])/(self.utility_cs*(self.dims[0])*self.utility_rho[i, 0]) + self.U[i, 0]*self.hx_area*(initial_fuel_Temps[i, 0] - initial_utility_Temps[i, 0])/(self.utility_cp[i, 0]*(self.utility_cs*(self.dims[0])*self.utility_rho[i,0])))

        #set initial row (fuel channel 1)
        for i in range(1, columns):
            dTdt_fuel[0, i] = (self.fuel_in[1]/self.dims[2]*(initial_fuel_Temps[0, i-1] - initial_fuel_Temps[0, i])/(self.fuel_cs*(self.dims[1])*self.fuel_rho[0, i]) - self.U[0, i]*self.hx_area*(initial_fuel_Temps[0, i] - initial_utility_Temps[0, i])/(self.fuel_cp[0, i]*(self.fuel_cs*(self.dims[1])*self.fuel_rho[0, i])))
            dTdt_utility[0, i] =  (self.utility_in[1]/self.dims[3]*(self.utility_in[2] - initial_utility_Temps[0, i])/(self.utility_cs*(self.dims[0])*self.utility_rho[0, i]) + self.U[0, i]*self.hx_area*(initial_fuel_Temps[0, i] - initial_utility_Temps[0, i])/(self.utility_cp[0, i]*(self.utility_cs*(self.dims[0])*self.utility_rho[0, i])))
        
        #solve remainder of arrays
        for j in range(1, columns):
            for i in range(1, rows):
                dTdt_fuel[i, j] = (self.fuel_in[1]/self.dims[2]*(initial_fuel_Temps[i, j-1] - initial_fuel_Temps[i, j])/(self.fuel_cs*(self.dims[1])*self.fuel_rho[i, j]) - self.U[i, j]*self.hx_area*(initial_fuel_Temps[i, j] - initial_utility_Temps[i, j])/(self.fuel_cp[i,j]*(self.fuel_cs*(self.dims[1])*self.fuel_rho[i, j])))
                dTdt_utility[i, j] =  (self.utility_in[1]/self.dims[3]*(initial_utility_Temps[i-1, j] - initial_utility_Temps[i, j])/(self.utility_cs*(self.dims[0])*self.utility_rho[i, j]) + self.U[i, j]*self.hx_area*(initial_fuel_Temps[i, j] - initial_utility_Temps[i, j])/(self.utility_cp[i, j]*(self.utility_cs*(self.dims[0])*self.utility_rho[i, j])))
                
        #wrap arrays up as a vector and return                
        dTdt = np.concatenate([dTdt_fuel.ravel(), dTdt_utility.ravel()])
        return dTdt

#set input dimensions and streams    
fuelin = [{'O2':5, 'CO2':92, 'H2O':3, 'CH4':0}, 0.001, 1000, 1500000]
utilityin = [{'H2O': 100}, 0.0005, 650, 1000000]
dimensions = [0.002, 0.001, 25, 50, 0.0002]

#solve steady state model
hx1 = crossflow_hx(fuelin, utilityin, dimensions)
fuelout, utilityout = hx1.solvehx()

#solve transient model - grab initial feed temperatures and use as input temperature
fuelinitials = fuelin[2]*np.ones(shape = (dimensions[2], dimensions[3]))
utilityinitials = utilityin[2]*np.ones(shape = (dimensions[2], dimensions[3]))

#use this to take SS model output as initial conditions
#initialTemps = np.concatenate([fuelout[0].ravel(), utilityout[0].ravel()])

#use this to take initial input conditions as initial conditions
initialTemps = np.concatenate([fuelinitials.ravel(), utilityinitials.ravel()])

#solve using an implicit solver (BDF)
sol = solve_ivp(hx1.transientHX, [0, 0.05], initialTemps, method = "BDF", t_eval = [0, 0.04, 0.05])

#grab final temperature profiles and reshape into given form
fuel_T_out = sol['y'][0:(dimensions[2]*(dimensions[3]))]
fuel_T_out = fuel_T_out[:, 2]
fuel_T_out = fuel_T_out.reshape(dimensions[2], dimensions[3])

util_T_out = sol['y'][(dimensions[2]*(dimensions[3])):]
util_T_out = util_T_out[:, 2]
util_T_out = util_T_out.reshape(dimensions[2], dimensions[3])
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 5 12:57:19 2021

@author: afurlong

Reduced order model for a printed circuit heat excahnger in crossflow
"""

#import necessary libraries
#import cantera as ct
import numpy as np
import math
from scipy.integrate import solve_ivp
import time
#import sys
#from scipy.optimize import minimize
#from scipy.optimize import Bounds
from scipy.optimize import fsolve
from scipy.optimize import root
#import matplotlib.pyplot as plt

class crossflow_PCHE(object):
    """
    Crossflow heat exchanger model for a single set of printed circuit heat exchanger (PCHE) plates.
    
    Parameters
    -----
    reactant_in : list of four elements
        0. reactant composition, dict of compositions, units of mass%
        1. mass flow rate through reactant plate, units of kg/s 
        2. reactant inlet temperature, units K        
        3. reactant inlet absolute pressure, units of Pa
    
    utility_in : list of four elements
        0. utility composition, dict of compositions, units of mass%        
        1. mass flow rate through utility plate, units of kg/s         
        2. utility inlet temperature, units K        
        3. utility inlet absolute pressure, units of Pa
        
    dims : list of five elements
        0. reactant channel diameter, units of m
        1. utility channel diameter, units of m
        2. number of reactant channels, dimensionless
        3. number of utility channels, dimensionless
        4. wall thickness between channels, units of m
        5. plate thickness, units of m
        
        note : fuel plate geometry assumed to be identical to reactant plate geometry
            spacing between utiltiy channels assumed to be identical to spacing between reactant channels
        
        
    Returns
    -----
    N/A at presennt

    Reference
    -----
    See other methods in class.
    
    Applicability
    -----
    Applicable for laminar flow (Re <= 2300) OR turbulent flow (Re > 2300). Transitional flow is not considered. Flow is treated as either all developed or all transient.
    
    Not suitable for use beyond approximately ideal gas conditions.
            
    """
    
    def __init__(self, reactant, utility, dimensions):
        self.reactant = reactant
        self.utility = utility
        self.dimensions = dimensions
        
        #grab #rows/#channels
        self.rows = self.dimensions[2]
        self.columns = self.dimensions[3]
        
        #set dimensions - cross sectional area
        self.reactant_cs = math.pi*self.dimensions[0]**2/8
        self.utility_cs = math.pi*self.dimensions[1]**2/8
        self.reactant_sqrtA = self.reactant_cs**0.5
        self.utility_sqrtA = self.utility_cs**0.5
        
        #aspect ratio for semicircular channels
        self.aspectratio = 0.5
        
        #set delta x/y/z to limit other calculation
        self.deltax = self.dimensions[1] + self.dimensions[4]
        self.deltay = self.dimensions[5]
        self.deltaz = self.dimensions[0] + self.dimensions[4]
        
        #define unit cell dimensions
        self.reactant_Vcell = self.reactant_cs*self.deltax
        self.utility_Vcell = self.utility_cs*self.deltaz
        self.reactantPlate_Vcell = self.deltax*self.deltay*self.deltaz - self.reactant_Vcell
        self.utilityPlate_Vcell = self.deltax*self.deltay*self.deltaz - self.utility_Vcell
        
        #define initial pressure profile (constant value)
        self.reactant_P = np.ones((self.rows, self.columns))*self.reactant[3]
        self.utility_P = np.ones((self.rows, self.columns))*self.utility[3]
    
        #define heat transfer areas
        #r - reactant, u - utility, f - fuel
        #P - plate, F - fluid
        self.hx_area_uPuF = (math.pi*self.dimensions[1]/2)*self.deltaz
        self.hx_area_uPrF = self.dimensions[0]*self.deltax
        self.hx_area_uPrP = self.deltax*self.deltaz - self.hx_area_uPrF
        self.hx_area_rPuP = self.deltax*self.deltaz - self.hx_area_uPrF
        self.hx_area_rPrF = (math.pi*self.dimensions[0]/2)*self.deltax
        self.hx_area_boundary = self.deltax*self.deltaz-self.dimensions[1]*self.deltax
        self.hx_area_boundary_solid = self.deltax*self.deltaz-self.hx_area_boundary
        
        #heat transfer areas within plates
        self.hx_area_rP_x = self.deltay*self.deltaz - self.reactant_cs
        self.hx_area_rP_z = self.deltax*self.deltay
        self.hx_area_uP_x = self.deltay*self.deltaz
        self.hx_area_uP_z = self.deltax*self.deltay - self.utility_cs
        
        #solid phase properties
        self.metalrho = 8000
        self.metalcp = 500
        self.metalk = 50
        
        #setup for initial viscosity parameters
        self.MW_list = {'H2': 2.016, 'Air': 28.964, 'N2': 28.013, 'O2': 31.999, 
                   'CO':28.010, 'CO2': 44.010, 'NO': 30.006, 'N2O': 44.012, 
                   'SO2': 64.065, 'CH4': 16.04, 'C2H2': 26.04, 'C2H4': 28.05, 
                   'C2H6':30.07, 'H2O': 18.02}
        
        self.sigma_list = {'H2': 2.915, 'Air': 3.617, 'N2': 3.667, 'O2': 3.433, 
                      'CO':3.590, 'CO2': 3.996, 'NO': 3.470, 'N2O': 3.879, 
                      'SO2': 4.026, 'CH4': 3.780, 'C2H2': 4.114, 'C2H4': 4.228,
                      'C2H6': 4.388, 'H2O': 2.725}
        
        self.epsOverKappa_list = {'H2': 38.0, 'Air': 97.0, 'N2': 99.8, 'O2': 113, 
                             'CO':110, 'CO2': 190, 'NO': 119, 'N2O': 220, 
                             'SO2': 363, 'CH4': 154, 'C2H2': 212, 'C2H4': 216, 
                             'C2H6': 232, 'H2O': 358.38}
        
        self.cp_a_list = {'H2': 3.249, 'Air': 3.355, 'N2': 3.280, 'O2': 3.639, 
                             'CO':3.376, 'CO2': 5.457, 'NO': 3.387, 'N2O': 5.328, 
                             'SO2': 5.699, 'CH4': 1.702, 'C2H2': 6.132, 
                             'C2H4': 1.424, 'C2H6': 1.131, 'H2O': 3.470}
        
        self.cp_b_list = {'H2': 0.422/1000, 'Air': 0.575/1000, 'N2': 0.593/1000, 
                          'O2': 0.506/1000, 'CO':0.557/1000, 'CO2': 1.045/1000,
                          'NO': 0.629/1000, 'N2O': 1.214/1000, 'SO2': 0.801/1000, 
                          'CH4': 9.081/1000, 'C2H2': 1.952/1000, 'C2H4': 14.394/1000, 
                          'C2H6': 19.225/1000, 'H2O': 1.450/1000}
        
        self.cp_c_list = {'H2': 0, 'Air': 0, 'N2': 0, 'O2': 0, 'CO':0, 'CO2': 0,
                          'NO': 0, 'N2O': 0, 'SO2': 0, 'CH4': -2.164/1000000, 
                          'C2H2': 0, 'C2H4': -4.392/1000000, 'C2H6': -5.561/1000000, 
                          'H2O': 0}
        
        self.cp_d_list = {'H2': 0.083*100000, 'Air': -0.016*100000, 'N2': 0.040*100000, 
                          'O2': -0.227*100000, 'CO':-0.031*100000, 'CO2': -1.157*100000, 
                          'NO': 0.014*100000, 'N2O': -0.928*100000, 'SO2': -1.015*100000, 
                          'CH4': 0, 'C2H2': -1.299*100000, 'C2H4': 0, 'C2H6': 0, 
                          'H2O': 0.121*100000}
        
        #ideal gas constant - J mol-1 K-1
        self.GC = 8.3144626
        
        #create arrays to store speicifc heat capacities
        self.cp_reactant, self.cp_utility= map(np.copy, [np.zeros((self.dimensions[2], self.dimensions[3]))]*2)
        
        self.mol_frac_and_cp()
        
        #set up positions for dimensionless length correlations
        #use center of each unit cell
        self.reactant_L = np.linspace(start = ((self.deltax)/2), 
                                      stop = (self.deltax*self.columns + (self.deltax)/2),
                                      num = self.columns)
        self.utility_L = np.linspace(start = ((self.deltaz)/2), 
                                      stop = (self.deltaz*self.rows + (self.deltaz)/2),
                                      num = self.rows)
        
        #constant for heat transfer correlations to avoid setting multiple times
        self.C1 = 3.24 #uniform wall temperature
        self.C2 = 1 #local
        self.C3 = 0.409 #uniform wall temperature
        self.C4 = 1 #local
        self.gamma = -0.1 #midpoint taken
        
    def update_reactant(self, reactant):
        #replace the reactant - use to set new conditions like flow rate, temperature, or pressure
        self.reactant = reactant
        self.mol_frac_and_cp()
        return
        
    def update_utility(self, utility):
        self.utility = utility
        self.mol_frac_and_cp()
        return
    
    def mol_frac_and_cp(self):
        #use molecular weights in init to convert
        #only convert when updated
        #only update coefficients for heat capacities when compositions are updates
        reactant_species = [*self.reactant[0]]
        utility_species = [*self.utility[0]]
        nmol_reactant = np.zeros(len(reactant_species))
        nmol_utility = np.zeros(len(utility_species))
        
        #reset constants for specific heat capacities
        self.cp_a_reactant = 0
        self.cp_b_reactant = 0
        self.cp_c_reactant = 0
        self.cp_d_reactant = 0
        self.cp_a_utility = 0
        self.cp_b_utility = 0
        self.cp_c_utility = 0
        self.cp_d_utility = 0        

        
        for i in range(len(reactant_species)):
            nmol_reactant[i] = self.reactant[0][reactant_species[i]]/self.MW_list[reactant_species[i]]
            self.cp_a_reactant = self.cp_a_reactant + nmol_reactant[i]*self.cp_a_list[reactant_species[i]]
            self.cp_b_reactant = self.cp_b_reactant + nmol_reactant[i]*self.cp_b_list[reactant_species[i]]
            self.cp_c_reactant = self.cp_c_reactant + nmol_reactant[i]*self.cp_c_list[reactant_species[i]]
            self.cp_d_reactant = self.cp_d_reactant + nmol_reactant[i]*self.cp_d_list[reactant_species[i]]
        for i in range(len(utility_species)):
            nmol_utility[i] = self.utility[0][utility_species[i]]/self.MW_list[utility_species[i]]
            self.cp_a_utility = self.cp_a_utility + nmol_utility[i]*self.cp_a_list[utility_species[i]]
            self.cp_b_utility = self.cp_b_utility + nmol_utility[i]*self.cp_b_list[utility_species[i]]
            self.cp_c_utility = self.cp_c_utility + nmol_utility[i]*self.cp_c_list[utility_species[i]]
            self.cp_d_utility = self.cp_d_utility + nmol_utility[i]*self.cp_d_list[utility_species[i]]

        self.cp_a_reactant = self.cp_a_reactant/nmol_reactant.sum()
        self.cp_b_reactant = self.cp_b_reactant/nmol_reactant.sum()
        self.cp_c_reactant = self.cp_c_reactant/nmol_reactant.sum()
        self.cp_d_reactant = self.cp_d_reactant/nmol_reactant.sum()
        self.cp_a_utility = self.cp_a_utility/nmol_utility.sum()
        self.cp_b_utility = self.cp_b_utility/nmol_utility.sum()
        self.cp_c_utility = self.cp_c_utility/nmol_utility.sum()
        self.cp_d_utility = self.cp_d_utility/nmol_utility.sum()
        
        reactant_molefrac = np.divide(nmol_reactant, nmol_reactant.sum())
        utility_molefrac = np.divide(nmol_utility, nmol_utility.sum())
            
        self.reactant_molefrac = dict(zip(reactant_species, reactant_molefrac))
        self.utility_molefrac = dict(zip(utility_species, utility_molefrac))
        
        self.reactant_MW = 0
        self.utility_MW = 0
        
        for i in range(len(reactant_species)):
            self.reactant_MW = self.reactant_MW + reactant_molefrac[i]*self.MW_list[reactant_species[i]]
        for i in range(len(utility_species)):
            self.utility_MW = self.utility_MW + utility_molefrac[i]*self.MW_list[utility_species[i]]

    def ff_Nu(self, fluid):
        """
        Evaluates the friction factors and Nusselt numbers for laminar and 
        turbulent flow for a given fluid

        Parameters
        ----------
        fluid : string
            String containing 'reactant', 'utility', or 'fuel' to pull class 
            variables for the given fluid

        Returns
        -------
        frictionfactor : array
            Array of floats containing friction factors for the given fluid in 
            laminar or turbulent flow
        nusselt : array
            Array of floats containing Nusself numbers for the given luid in
            laminar or turbulent flow
        
        Applicability
        -----
        Applicable for developing laminar flow (Re <= 2300), or fully developed
        turbulent flow in a smooth channel (Re > 2300). Transitional flow is 
        not considered.
        
        Reference
        ------
        Muzychka, Y. S., & Yovanovich, M. M. (2009). Pressure Drop in Laminar 
        Developing Flow in Noncircular Ducts: A Scaling and Modeling Approach. 
        Journal of Fluids Engineering, 131(11). 
        https://doi.org/10.1115/1.4000377
        
        Muzychka, Y. S., & Yovanovich, M. M. (2004). Laminar Forced Convection 
        Heat Transfer in the Combined Entry Region of Non-Circular Ducts. 
        Journal of Heat Transfer, 126(1), 54–61. 
        https://doi.org/10.1115/1.1643752
        
        Gnielinski correlation for turbulent flow

        """
        
        #evaluate L+ for each fluid (dimensionless position for use in friction factor correlations)
        if fluid == 'reactant':
            Lplus = self.reactant_mu*self.reactant_L/(self.reactant[1]/self.rows)
            reynolds = self.reactant_Re
            
            Pr = self.reactant_Pr
            zstar = (self.reactant_L/self.reactant_sqrtA)/(reynolds*Pr)
            
        elif fluid == 'utility':
            Lplus = (self.utility_mu.transpose()*self.utility_L).transpose()/(self.utility[1]/self.columns)
            reynolds = self.utility_Re
            
            Pr = self.utility_Pr
            zstar = ((self.utility_L/self.reactant_sqrtA)/((reynolds*Pr).transpose())).transpose()
                 
        else:
            print('Incorrect fluid selected for friction factor!')
        
        m = 2.27 + 1.65*Pr**(1/3)
        fPr = 0.564/((1+(1.664*Pr**(1/6))**(9/2))**(2/9))
        
        laminar = np.less_equal(reynolds, 2300)
        turbulent = np.greater(reynolds, 2300)
        
        laminar_f = ((3.44 * Lplus**-0.5)**2 + (12 / (self.aspectratio**0.5 * (1 + self.aspectratio) * (1 - 192*self.aspectratio * math.pi**-5 * math.tanh(math.pi / (2*self.aspectratio)))))**2)**0.5/reynolds
        turbulent_f = (0.79*np.log(reynolds) - 1.64)**-2/8
        frictionfactor = laminar*laminar_f + turbulent*turbulent_f
        
        #this might need np.power instead of exponents
        nusselt_laminar = ((self.C4*fPr/zstar**0.5)**m + ((self.C2*self.C3*(laminar_f*reynolds/zstar)**(1/3))**5 + (self.C1*(laminar_f*reynolds/(8*math.pi**0.5*self.aspectratio**self.gamma)))**5)**(m/5))**(1/m)
        nusselt_turbulent = (turbulent_f*(reynolds-1000)*Pr)/(1+12.7*turbulent_f**0.5 * (Pr**(2/3) - 1))
        nusselt = laminar*nusselt_laminar + turbulent*nusselt_turbulent
        
        # if np.isnan(nusselt).any() == True or np.isnan(frictionfactor).any() == True:
        #     print('Nusselt/FF failed')
        #     #sys.exit()
        
        return frictionfactor, nusselt
        
    def properties(self, fluid):
        """
        Evaluate the viscosity and thermal conductivity of an ideal gas as a f
        unction of temperature using Wilke's approach given in Bird, Stewart, 
        and Lightfoot.
        
        Best suited for a minimal number of species (i.e. eliminate minor 
        species present) as the time to solve is exponentially reliant on the 
        number of species for interaction parameters.

        Parameters
        ----------
        fluid : String
            Fluid name, either 'reactant', 'utility', or 'fuel'. Uses this 
            information to select the correct set of temperatures and 
            compositions.

        Returns
        -------
        viscosity_mixture : Array, float
            Viscosity of the given fluid, units of Pa.s.
            
        k_mixture : Array, float
            Thermal conductivity of the fluid, units of W m-1 K-1.
            
        Applicability
        -----
        Applicable for small molecules, set for molecules up to ethane. Other 
        species can be added to the list in object initialization.
        
        Reference
        -----
        Bird, R. B., Stewart, W. E., & Lightfoot, E. N. (1966). 
        Transport Phenomena. Brisbane, QLD, Australia: 
        John Wiley and Sons (WIE). - note 2001 edition used for interaction
        parameters. Methodology for viscosity from Chapter 1.4, thermal 
        conductivity from Chapter 9.3, tabulated data from Appendix E.1.
        
        """
        #grab fluid composition, update cp
        if fluid == 'reactant':
            composition = self.reactant[0]
            molfractions = self.reactant_molefrac
            temperatures = self.reactant_T
            self.reactant_rho = self.reactant_P*self.reactant_MW/self.GC/self.reactant_T/1000
            self.reactant_cp = self.cp_a_reactant + self.cp_b_reactant*temperatures + self.cp_c_reactant*np.power(temperatures, 2) + self.cp_d_reactant*np.power(temperatures, -2)
            self.reactant_cp = self.reactant_cp*self.GC/self.reactant_MW*1000 #to J/mol K, to J/kg K
            
        elif fluid == 'utility':
            composition = self.utility[0]
            molfractions = self.utility_molefrac
            temperatures = self.utility_T
            self.utility_rho = self.utility_P*self.utility_MW/self.GC/self.utility_T/1000
            self.utility_cp = self.cp_a_utility + self.cp_b_utility*temperatures + self.cp_c_utility*np.power(temperatures, 2) + self.cp_d_utility*np.power(temperatures, -2)
            self.utility_cp = self.utility_cp*self.GC/self.utility_MW*1000 #to J/mol K, to J/kg K

        else:
            print('Incorrect fluid selected for properties update!')
            
        #extract list of species
        species = [*composition]
        nspecies = len(species)
        
        #create arrays for fluids and intermediate calculations
        viscosity, Tstar, omega, ki, cpi = map(np.copy, [np.zeros((self.dimensions[2], self.dimensions[3], nspecies))]*5)
        phi = np.zeros((self.dimensions[2], self.dimensions[3], nspecies**2))
        viscosity_mixture, k_mixture = map(np.copy, [np.zeros((self.dimensions[2], self.dimensions[3]))]*2)
        
        # if temperatures.min() < 0:
        #     print('temperatures < 0, properties failed')
            #sys.exit()
        
        #calculate values for arrays
        for i in range(nspecies):
            Tstar[:, :, i] = temperatures/self.epsOverKappa_list[species[i]]
            omega[:, :, i] = 1.16145/(np.power(Tstar[:, :, i], 0.14874)) + 0.52487/(np.exp(0.77320*Tstar[:, :, i])) + 2.16178/(np.exp(2.43787*Tstar[:, :, i]))
            viscosity[:, :, i] = (2.6693 * (10**(-5)) * np.sqrt(self.MW_list[species[i]]*temperatures) / (self.sigma_list[species[i]]**2 * omega[:, :, i]))*98.0665/1000

        for i in range(nspecies):
            cpi[:, :, i] = self.cp_a_list[species[i]] + self.cp_b_list[species[i]]*temperatures + self.cp_c_list[species[i]]*np.power(temperatures, 2) + self.cp_d_list[species[i]]*np.power(temperatures, -2)
            ki[:, :, i] = (cpi[:, :, i] + 5/4)*8314.4626*viscosity[:, :, i]/self.MW_list[species[i]]
                    
        #calculate values of phi for each interaction -- use 3D array with one 2D array for every pair
        #this is made exponentially slower for every added species
        for i in range(nspecies):
            for j in range(nspecies):
                phi[:, :, nspecies*i + j] = 1/(8**0.5) * (1 + self.MW_list[species[i]]/self.MW_list[species[j]])**-0.5 * (1 + (viscosity[:, :, i]/viscosity[:, :, j])**0.5 * (self.MW_list[species[j]]/self.MW_list[species[i]])**0.25)**2

        #apply mixing rules
        for i in  range(nspecies):
            denominator = 0
            for j in range(nspecies):
                denominator = denominator + molfractions[species[j]]*phi[:, :, nspecies*i + j]
            viscosity_mixture = viscosity_mixture + molfractions[species[i]]*viscosity[:, :, i]/denominator
            k_mixture = k_mixture + molfractions[species[i]]*ki[:, :, i]/denominator
        
        return viscosity_mixture, k_mixture
    
    def unwrap_T(self, Tvector):
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
            2-Dimensional reactant temperature profile. Units of K.
        initial_utility_Temps : Array
            2-Dimensional utility temperature profile. Units of K.
        initial_fuel_Temps : Array
            2-Dimensional fuel temperature profile. Units of K.
        initial_reactantPlate_Temps : Array
            2-Dimensional reactant plate temperature profile. Units of K.
        initial_utilityPlate_Temps : Array
            2-Dimensional utility plate temperature profile. Units of K.
        initial_fuelPlate_Temps : Array
            2-Dimensional fuel plate temperature profile. Units of K.
        """
        initial_reactant_Temps = Tvector[0:self.rows*self.columns].reshape(self.rows, self.columns)
        initial_utility_Temps = Tvector[self.rows*self.columns:self.rows*self.columns*2].reshape(self.rows, self.columns)
        initial_reactantPlate_Temps = Tvector[self.rows*self.columns*2:self.rows*self.columns*3].reshape(self.rows, self.columns)
        initial_utilityPlate_Temps = Tvector[self.rows*self.columns*3:self.rows*self.columns*4].reshape(self.rows, self.columns)
        
        return initial_reactant_Temps, initial_utility_Temps, initial_reactantPlate_Temps, initial_utilityPlate_Temps
    
    def intraplate_cond(self, plate):
        """
        #######################################################################
        THIS SECTION NEEDS VALIDATION OF RESULTS/IS GIVING ABNORMAL RESULTS
        #######################################################################
        
        Offset the temperature in the x/z axis to handle conduction within plates.
        Uses numpy's roll with the first/last row or column as an insulated boundary.
        
         Parameters
        ----------
        plate : string
            String consisting of either 'reactant', 'utility', or 'fuel' for the
            selection of a plate.
            
        Returns
        -------
        Qnet : Array
            Net heat transfer in/out of each cell in the plate
        """
        
        if plate == 'reactant':
            temps = self.reactantPlate_T
        elif plate == 'utility':
            temps = self.utilityPlate_T
        else:
            print('Incorrect plate selected for conduction terms!!')    

        offset_x_fwd = np.roll(temps, 1, 1)
        offset_x_rev = np.roll(temps, -1, 1)
        offset_z_fwd = np.roll(temps, 1, 0)
        offset_z_rev = np.roll(temps, -1, 0)
        
        temp_x_dir = (offset_x_rev - 2*temps + offset_x_fwd)/(self.deltax**2)
        temp_x_dir[:, 0] = 2*(temps[:, 1] - temps[:, 0])/(self.deltax**2)
        temp_x_dir[:, -1] = 2*(temps[:, -2] - temps[:, -1])/(self.deltax**2)
        
        temp_z_dir = (offset_z_rev - 2*temps + offset_z_fwd)/(self.deltaz**2)
        temp_z_dir[0, :] = 2*(temps[1, :] - temps[0, :])/(self.deltaz**2)
        temp_z_dir[-1, :] = 2*(temps[-2, :] - temps[-1, :])/(self.deltaz**2)
                
        dTdtNet = (temp_x_dir + temp_z_dir)*(self.metalk/(self.metalrho*self.metalcp))
        return dTdtNet
    
    def advective_transfer(self, fluid):
        """
        Determine the advective heat transfer terms

        Parameters
        ----------
        fluid : String
            String containing 'reactant', 'utility', or 'fuel' to set the 
            direction and temperature profile for heat transfer.

        Returns
        -------
        deltaT : Array
            Array of differential temperature terms to add to transient solver

        """
        
        if fluid == 'reactant':
            temps = self.reactant_T
            roll_dir = 1
            T0 = self.reactant[2]
            
        elif fluid == 'utility':
            temps = self.utility_T
            roll_dir = 0
            T0 = self.utility[2]
            
        else: 
            print('Incorrect fluid selected for advective term!!')
        
        #offset temperature to get upstream temperature in each location
        offset_T = np.roll(temps, 1, roll_dir)
        deltaT = offset_T - temps
        
        
        #add boundary condition
        if roll_dir == 0:
            deltaT[0, :] = T0 - temps[0, :]
        elif roll_dir == 1:
            deltaT[:, 0] = T0 - temps[:, 0]
            
        return deltaT
    
    def update_pressures(self):
        """
        Update the pressure profile through the heat exchanger with Bernoulli's
        equation. D_h = square root of cross-sectional area
        
        Parameters
        -------
        None - pulled from class.

        Returns
        -------
        None - stored in class.

        """
        
        #calculate losses via Bernoulli's equation
        deltaP_reactant = 2*self.reactant_f*self.deltax/self.reactant_sqrtA*self.reactant_u**2*self.reactant_rho
        deltaP_utility = 2*self.utility_f*self.deltaz/self.utility_sqrtA*self.utility_u**2*self.utility_rho
        #print(deltaP_reactant)
    
        #if the array is unchanged from its initial state, then initialize
        if self.reactant_P.mean() == self.reactant_P[0, 0]:
            self.reactant_P[:, 0] = self.reactant[3] - deltaP_reactant[:, 0]
            self.utility_P[0, :] = self.utility[3] - deltaP_utility[0, :]
            
            for i in range(1, self.columns):
                self.reactant_P[:, i] = self.reactant_P[:, i-1] - deltaP_reactant[:, i]
                
            for i in range(1, self.rows):
                self.utility_P[i, :] = self.utility_P[i-1, :] - deltaP_utility[i, :]
            
        #if the pressure drop has been initialized before, then update it with the new delta P    
        else:
            self.reactant_P = np.roll(self.reactant_P, 1, 1) - deltaP_reactant
            self.reactant_P[:, 0] = self.reactant[3] - deltaP_reactant[:, 0]
            self.utility_P = np.roll(self.utility_P, 0, 1) - deltaP_utility
            self.utility_P[0, :] = self.utility[3] - deltaP_utility[0, :]
            
        print(self.reactant_P)
    
        return
    
    def transient_solver(self, t, T):
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
        
        #start with extracting the initial temperature profile, setting dTdt to 0.
        self.reactant_T, self.utility_T, self.reactantPlate_T, self.utilityPlate_T = self.unwrap_T(T)
        dTdt_reactant, dTdt_utility, dTdt_reactantPlate, dTdt_utilityPlate = map(np.copy, [np.zeros((self.rows, self.columns))]*4)
        
        #print(self.utility_T)

        
        #update properties for the class
        self.reactant_mu, self.reactant_k = self.properties('reactant')
        self.utility_mu, self.utility_k = self.properties('utility')
        
        #calculate bulk velocity in channels
        self.reactant_u = self.reactant[1]/(self.reactant_rho*self.dimensions[2]*self.reactant_cs)
        self.utility_u = self.utility[1]/(self.utility_rho*self.dimensions[3]*self.utility_cs)
        
        #update Prandtl number, Reynolds number
        self.reactant_Pr = self.reactant_mu*self.reactant_cp/self.reactant_k
        self.utility_Pr = self.utility_mu*self.utility_cp/self.utility_k
        
        self.reactant_Re = self.reactant_rho*self.reactant_u*self.reactant_sqrtA/self.reactant_mu
        self.utility_Re = self.utility_rho*self.utility_u*self.utility_sqrtA/self.utility_mu
        
        #update friction factors
        self.reactant_f, self.reactant_Nu = self.ff_Nu('reactant')
        self.utility_f, self.utility_Nu = self.ff_Nu('utility')
        
        #calculate convective heat transfer coefficients
        self.reactant_h = self.reactant_Nu*self.reactant_k/self.reactant_sqrtA
        self.utility_h = self.utility_Nu*self.utility_k/self.utility_sqrtA
        
        #calculate heat transfer between fluids and plates
        #positive value = heat gained by CV
        #negative value= heat lost by CV
        
        #convective transfer for fluids
        self.Q_reactant_fluid = self.reactant_h*(self.hx_area_uPrF*(self.utilityPlate_T - self.reactant_T) + self.hx_area_rPrF*(self.reactantPlate_T - self.reactant_T))
        self.Q_utility_fluid = self.utility_h*(self.hx_area_uPuF*(self.utilityPlate_T - self.utility_T) + self.hx_area_boundary*(self.reactantPlate_T - self.utility_T))
        
        #convective transfer to solids
        self.Q_reactant_plate = self.reactant_h*self.hx_area_rPrF*(self.reactant_T - self.reactantPlate_T) + self.utility_h*self.hx_area_boundary*(self.utility_T - self.reactantPlate_T)
        self.Q_utility_plate = self.utility_h*self.hx_area_uPuF*(self.utility_T - self.utilityPlate_T) + self.reactant_h*self.hx_area_uPrF*(self.reactant_T - self.utilityPlate_T)
        
        #conductive transfer between plates (interplate/y-direction)
        #this will need to be updated following evaluation in 3D as no shape factor data is available
        self.Q_reactant_plate = self.Q_reactant_plate + self.metalk*(self.hx_area_rPuP*(self.utilityPlate_T - self.reactantPlate_T) + self.hx_area_boundary_solid*(self.utilityPlate_T - self.reactantPlate_T))/self.deltay
        self.Q_utility_plate = self.Q_utility_plate + self.metalk*(self.hx_area_uPrP*(self.reactantPlate_T - self.utilityPlate_T) + self.hx_area_boundary_solid*(self.reactantPlate_T - self.utilityPlate_T))/self.deltay
        
        #add intraplate conduction terms
        #conduction perpendicular to flow needs to be updated with shape factors
        # self.Q_reactant_plate = self.Q_reactant_plate + self.intraplate_cond('reactant')
        # self.Q_utility_plate = self.Q_utility_plate + self.intraplate_cond('utility')
        
        #totalQ = self.Q_reactant_plate + self.Q_utility_plate + self.Q_reactant_fluid + self.Q_utility_fluid
        #print(totalQ.sum())
        
        #print(self.reactantPlate_T)
        
        #convert from heat transfer to dT, neglective advective term
        dTdt_reactant = self.Q_reactant_fluid/(self.reactant_rho*self.reactant_Vcell*self.reactant_cp)
        dTdt_utility = self.Q_utility_fluid/(self.utility_rho*self.utility_Vcell*self.utility_cp)
        
        dTdt_reactantPlate = self.Q_reactant_plate/(self.metalrho*self.reactantPlate_Vcell*self.metalcp)
        dTdt_utilityPlate = self.Q_utility_plate/(self.metalrho*self.utilityPlate_Vcell*self.metalcp)
        
        #add the advective term to dTdt
        advective_reactant = self.reactant_u*self.advective_transfer('reactant')/self.deltax
        advective_utility = self.utility_u*self.advective_transfer('utility')/self.deltaz        
        dTdt_reactant = dTdt_reactant + advective_reactant
        dTdt_utility = dTdt_utility + advective_utility
        
        #wrap up dT/dt as a vector for use in solve ivp
        dTdt = np.concatenate([dTdt_reactant.ravel(), dTdt_utility.ravel(), 
                               dTdt_reactantPlate.ravel(), dTdt_utilityPlate.ravel()])

        return dTdt
    
    def steady_solver(self, initialTemps):
        dTdt = self.transient_solver(0, initialTemps)
        #dTdt = (dTdt**2).sum()**0.5 #comment out for fsolve
        #print(dTdt)
        return dTdt
        
###############################################################################
###############################################################################    

def convert_T_vector(T_vector, dims):
    reactantTemps = T_vector[0:dims[2]*dims[3]].reshape(dims[2], dims[3])
    utilityTemps = T_vector[dims[2]*dims[3]:dims[2]*dims[3]*2].reshape(dims[2], dims[3])
    reactantPlateTemps = T_vector[dims[2]*dims[3]*2:dims[2]*dims[3]*3].reshape(dims[2], dims[3])
    utilityPlateTemps = T_vector[dims[2]*dims[3]*3:dims[2]*dims[3]*4].reshape(dims[2], dims[3])
    
    return reactantTemps, utilityTemps, reactantPlateTemps, utilityPlateTemps


reactant_inlet = [{'CO2': 100, 'H2O': 0, 'O2': 0}, 0.00702, 1000, 1500000]
utility_inlet = [{'H2O': 100}, 0.005, 1000, 1000000]
dimensions = [0.0015, 0.0015, 20, 35, 0.0011, 0.0021]

exchanger = crossflow_PCHE(reactant_inlet, utility_inlet, dimensions)

initial_T_reactant = reactant_inlet[2]*np.ones((dimensions[2], dimensions[3]))
initial_T_reactantPlate = reactant_inlet[2]*np.ones((dimensions[2], dimensions[3]))
initial_T_utility = utility_inlet[2]*np.ones((dimensions[2], dimensions[3]))
initial_T_utilityPlate = utility_inlet[2]*np.ones((dimensions[2], dimensions[3]))

initial_temps = np.concatenate([initial_T_reactant.ravel(), initial_T_utility.ravel(),
                                initial_T_reactantPlate.ravel(), initial_T_utilityPlate.ravel()])

t0 = time.time()

for i in range(1):
    exchanger = crossflow_PCHE(reactant_inlet, utility_inlet, dimensions)
    solution = solve_ivp(exchanger.transient_solver, [0, 10000], initial_temps, method = 'BDF', t_eval = [0, 1, 10, 100, 1000, 10000])
tend = time.time()

print('time to solve to steady-state with BDF:', tend-t0, 's')

exchanger.update_pressures()

solution = solve_ivp(exchanger.transient_solver, [0, 10000], solution['y'][:, -1], method = 'BDF', t_eval = [0, 10000])

exchanger.update_pressures()

T_reactant, T_utility, T_reactant_plate, T_utility_plate = convert_T_vector(solution['y'][:, -1], dimensions)
P_reactant = exchanger.reactant_P.min()
P_utility = exchanger.utility_P.min()

# t1 = time.time()
# for i in range(1):
#     exchanger = crossflow_PCHE(reactant_inlet, utility_inlet, dimensions)
#     sol_fsolve = fsolve(exchanger.steady_solver, initial_temps)#initial_temps)
# tend = time.time()
# print('time to solve to steady-state with fsolve: ', tend-t1, 's')

# deltas_fsolve = abs((solution['y'][:, -1] - sol_fsolve))



#cons = LinearConstraint(initial_temps, utility_inlet[2], reactant_inlet[2])
#cons2 = {'type': 'ineq', 'fun': cons}
#bnds = Bounds(utility_inlet[2], reactant_inlet[2])

# sol_min = minimize(exchanger.steady_solver, 
#                    initial_temps, 
#                    method = 'SLSQP', 
#                    options = {'maxiter': 1000, 'disp': True},
#                    bounds = bnds)

#SLSQP: 49.95 resid, 687 iter, 278283 evals, 312.4s 
######### w/ eps = 1: high resid, 410201 evals, 660s
#trust-constr: 69.5 resid, 1000 iter, 564207 evals, 496.9s 

# t2 = time.time()
# for i in range(1):
#     exchanger = crossflow_PCHE(reactant_inlet, utility_inlet, dimensions)
#     sol_root = root(exchanger.steady_solver, initial_temps)
# tend = time.time()
# print('time to solve to steady-state with root: ', tend - t2, 's')
# deltas_root = abs((solution['y'][:, -1] - sol_root['x']))

# t3= time.time()
# for i in range(2):
#     exchanger = crossflow_PCHE(reactant_inlet, utility_inlet, dimensions)
#     sol_root_krylov = root(exchanger.steady_solver, initial_temps, method='Krylov')
# tend = time.time()
# print('time to solve to steady-state with root/Krylov: ', tend - t3, 's')
# deltas_root = abs((solution['y'][:, -1] - sol_root_krylov['x']))


# t4 = time.time()
# for i in range(1):
#     exchanger = crossflow_PCHE(reactant_inlet, utility_inlet, dimensions)
#     solution = solve_ivp(exchanger.transient_solver, [0, 10000], initial_temps, method = 'LSODA', t_eval = [0, 1, 10, 100, 1000, 10000])
# tend = time.time()

# print('time to solve to steady-state with LSODA:', tend-t4, 's')

# t5 = time.time()
# for i in range(2):
#     solution = solve_ivp(exchanger.transient_solver, [0, 10000], initial_temps, method = 'RK23', t_eval = [0, 1, 10, 100, 1000, 10000])
# tend = time.time()

# print('time to solve to steady-state with RK23:', tend-t5, 's')

# t6= time.time()
# for i in range(2):
#     exchanger = crossflow_PCHE(reactant_inlet, utility_inlet, dimensions)
#     solution = solve_ivp(exchanger.transient_solver, [0, 10000], initial_temps, method = 'RK45', t_eval = [0, 1, 10, 100, 1000, 10000])
# tend = time.time()

# print('time to solve to steady-state with RK23:', tend-t6, 's')

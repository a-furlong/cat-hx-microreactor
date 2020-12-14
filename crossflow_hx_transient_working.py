#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:52:05 2020
Modified on 23 Nov

Cross flow heat exchanger model function

author: afurlong
"""

import cantera as ct
import numpy as np
import math
import scipy.optimize as opt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class crossflow_hx(object):
    def __init__(self, fuel_in, steam_in, dims):
        self.fuel_in = fuel_in
        #fuel should be a list of [[fuel composition], fuel mass flow rate, fuel temperature in K, fuel pressure in Pa]
        self.steam_in = steam_in
        #steam should be a list of [[steam composition], steam mass flow rate, steam temperature in K, steam pressure in Pa]
        self.dims = dims
        #dimensions are [fuel channel height, fuel channel width, steam channel height, steam channel width, number fuel channels, number steam channels, wall thicknesses for channels]
        #dimensions are [fuel channel diameter, steam channel diameter, number fuel channels, number steam channels, wall thicknesses for channels]

        
        #calculate dimensions needed for basic calculations
        self.fuel_cs = self.dims[0]**2/8
        self.steam_cs = self.dims[1]**2/8
        
        self.fuel_dh = math.pi*self.dims[0]/(2*(1+math.pi/2))
        self.steam_dh = math.pi*self.dims[1]/(2*(1+math.pi/2))
        
        self.hx_area = self.dims[0]*self.dims[1]
        
        #self.fuel_eps = min(self.dims[0], self.dims[1])/max(self.dims[0], self.dims[1])
        #self.steam_eps = min(self.dims[2], self.dims[3])/max(self.dims[2], self.dims[3])
        self.fuel_eps = 0.5
        self.steam_eps = 0.5
        self.fuel_sqrtA = self.fuel_cs**0.5
        self.steam_sqrtA = self.steam_cs**0.5
        
        #set up initial cantera fluids - gri30 good for combustion of natural gas
        #not setting up initial fluid conditions here
        self.fuel = ct.Solution('gri30.xml')
        self.steam = ct.Solution('gri30.xml')
        self.fuel.transport_model = 'Mix'
        self.steam.transport_model = 'Mix'
           
        """
        set up inital arrays to reserve memory
        needed arrays are temperature, pressure, density, velocity, viscosity, 
        reynolds number, specific heat capacity, thermal conductivity, 
        prandtl number, nusselt number, convective heat transfer coefficient, 
        overall heat transfer coefficient, total heat transfer, z for hx calcs length
        """
        
        self.fuel_T = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_T = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_P = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_P = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_rho = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_rho = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_u = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_u = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_mu = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_mu = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_Re = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_Re = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_cp = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_cp = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_k = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_k = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_Pr = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_Pr = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_Nu = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_Nu = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.fuel_h = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_h = self.emptyarray(self.dims[2], self.dims[3], 2)
        self.U = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.Q = self.emptyarray(self.dims[2], self.dims[3], 1)
        
        self.fuel_z = np.linspace(start = 0, stop = self.dims[3]*(self.dims[1] + self.dims[4]) - self.dims[4], num = self.dims[3] + 1)
        self.steam_z = np.linspace(start = 0, stop = self.dims[2]*(self.dims[0] + self.dims[4]) - self.dims[4], num = self.dims[2] + 1)
        
        #setup for pressure drop
        self.fuel_lplus = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_lplus = self.emptyarray(self.dims[2], self.dims[3], 2)
        #print(self.steam_lplus)
        self.fuel_f = self.emptyarray(self.dims[2], self.dims[3], 1)
        self.steam_f = self.emptyarray(self.dims[2], self.dims[3], 2)  
        
    def solvehx(self):
        #set initial fluid conditions and arrays
        self.fuel.TPX = self.fuel_in[2], self.fuel_in[3], self.fuel_in[0]
        self.steam.TPX = self.steam_in[2], self.steam_in[3], self.steam_in[0]
        
        #ignore friction for now on pressure. assume no gas-phase reactions
        self.fuel_T[:, 0] = self.fuel_in[2]
        self.fuel_P[:, 0] = self.fuel_in[3]
        self.steam_T[0, :] = self.steam_in[2]
        self.steam_P[0, :] = self.steam_in[3]
        
        self.fuel_rho[:, 0] = self.fuel.density
        self.steam_rho[0, :] = self.steam.density
        self.fuel_u[:, 0] = self.fuel_in[1]/(self.fuel.density*self.dims[2]*self.fuel_cs)
        self.steam_u[0, :] = self.steam_in[1]/(self.steam.density*self.dims[3]*self.steam_cs)
        self.fuel_mu[:, 0] = self.fuel.viscosity
        self.steam_mu[0, :] = self.steam.viscosity
        self.fuel_Re[:, 0] = self.reynolds(self.fuel.density, self.fuel_u[0, 0], self.fuel_dh, self.fuel.viscosity)
        self.steam_Re[0, :] = self.reynolds(self.steam.density, self.steam_u[0, 0], self.steam_dh, self.steam.viscosity)
        self.fuel_cp[:, 0] = self.fuel.cp_mass
        self.steam_cp[0, :] = self.steam.cp_mass
        self.fuel_k[:, 0] = self.fuel.thermal_conductivity
        self.steam_k[0, :] = self.steam.thermal_conductivity
        self.fuel_Pr[:, 0] = self.fuel_mu[0, 0]*self.fuel_cp[0, 0]/self.fuel_k[0, 0]
        self.steam_Pr[0, :] = self.steam_mu[0, 0]*self.steam_cp[0, 0]/self.steam_k[0, 0]
        
        #Muzychka & Yovanovich - combined entry region for small z*
        #self.fuel_Nu[:, 0] = self.entryNuDh(0, self.fuel_dh, self.fuel_Re[0, 0], self.fuel_Pr[0, 0])
        #self.steam_Nu[0, :] = self.entryNuDh(0, self.steam_dh, self.steam_Re[0, 0], self.steam_Pr[0, 0])
        self.fuel_Nu[:, 0] = self.entryNuDh(0, self.fuel_dh, self.fuel_Re[0, 0], self.fuel_Pr[0, 0], self.fuel_eps, self.fuel_sqrtA)
        self.steam_Nu[0, :] = self.entryNuDh(0, self.steam_dh, self.steam_Re[0, 0], self.steam_Pr[0, 0], self.steam_eps, self.steam_sqrtA)
        self.fuel_h[:, 0] = self.fuel_Nu[0, 0] * self.fuel_k[0, 0]/self.fuel_dh
        self.steam_h[0, :] = self.steam_Nu[0, 0] * self.steam_k[0, 0]/self.steam_dh
        
        #Muzychka & Yovanovich - entry region friction factor
        #L+ is 0 for the entry region
        self.fuel_f[:, 0] = self.devflow_friction(self.fuel_lplus[0, 0], self.fuel_eps, self.fuel_Re[0, 0], self.fuel_dh, self.fuel_sqrtA)
        self.steam_f[0, :] = self.devflow_friction(self.steam_lplus[0, 0], self.steam_eps, self.steam_Re[0, 0], self.steam_dh, self.steam_sqrtA)
        
        self.steam_P[:, :] = self.steam_P[0, 0]
        
        #now iterate through and solve for everything
        for j in range(0, self.dims[3]):
            for i in range(0, self.dims[2]):
                #calculate heat tranfer coefficient and total heat transfer
                self.U[i, j] = 1/(1/self.fuel_h[i, j] + 1/self.steam_h[i, j] + 0.0005/45)
                self.Q[i, j] = self.U[i, j]*self.hx_area*(self.fuel_T[i, j] - self.steam_T[i, j])
                #self.fuel_T[i, j+1] = self.fuel_T[i, j] - self.Q[i, j]/(self.fuel_rho[i, j]*self.fuel_u[i, j]*self.fuel_cp[i, j]*self.fuel_cs)
                #self.steam_T[i+1, j] = self.steam_T[i, j] + self.Q[i, j]/(self.steam_rho[i, j]*self.steam_u[i, j]*self.steam_cp[i, j]*self.steam_cs)
                #use the mass flow rate through each channel instead of the velocity in each channel for the heat transferred
                self.fuel_T[i, j+1] = self.fuel_T[i, j] - self.Q[i, j]/(self.fuel_in[1]/self.dims[2]*self.fuel_cp[i, j])
                self.steam_T[i+1, j] = self.steam_T[i, j] + self.Q[i, j]/(self.steam_in[1]/self.dims[3]*self.steam_cp[i, j])
                
                #assuming incompressible flow given low velocity
                self.fuel_P[i, j+1] = self.fuel_P[i, j] - 2*self.fuel_f[i, j] * (self.dims[1] + self.dims[4]) / self.fuel_dh *self.fuel_rho[i, j]*self.fuel_u[i, j]**2
                self.steam_P[i+1, j] = self.steam_P[i, j] - 2*self.steam_f[i, j] * (self.dims[0] + self.dims[4]) / self.steam_dh *self.steam_rho[i, j]*self.steam_u[i, j]**2

                #update temperature and pressure for new cells
                self.fuel.TP = self.fuel_T[i, j+1], self.fuel_P[i, j+1]
                self.steam.TP = self.steam_T[i+1, j], self.steam_P[i+1, j]
        
                #update thermophysical and transport properties for new conditions
                self.fuel_rho[i, j+1] = self.fuel.density
                self.steam_rho[i+1, j] = self.steam.density
                self.fuel_u[i, j+1] = self.fuel_u[i, j]*self.fuel_rho[i, j]/self.fuel_rho[i, j+1]
                self.steam_u[i+1, j] = self.steam_u[i, j]*self.steam_rho[i, j]/self.steam_rho[i+1, j]
                self.fuel_mu[i, j+1] = self.fuel.viscosity
                self.steam_mu[i+1, j] = self.steam.viscosity
                self.fuel_cp[i, j+1] = self.fuel.cp_mass
                self.steam_cp[i+1, j] = self.steam.cp_mass
                self.fuel_k[i, j+1] = self.fuel.thermal_conductivity
                self.steam_k[i+1, j] = self.steam.thermal_conductivity
                self.fuel_Pr[i, j+1] = self.fuel_mu[i, j+1]*self.fuel_cp[i, j+1]/self.fuel_k[i, j+1]
                self.steam_Pr[i+1, j] = self.steam_mu[i+1, j]*self.steam_cp[i+1, j]/self.steam_k[i+1, j]
                
                self.fuel_Re[i, j+1] = self.reynolds(self.fuel_rho[i, j+1], self.fuel_u[i, j+1], self.fuel_dh, self.fuel_mu[i, j+1])
                self.steam_Re[i+1, j] = self.reynolds(self.steam_rho[i+1, j], self.steam_u[i+1, j], self.steam_dh, self.steam_mu[i+1, j])
                
                #determine developing region friction factors
                #self.fuel_Nu[i, j+1] = self.entryNuDh(self.fuel_z[j+1], self.fuel_dh, self.fuel_Re[i, j+1], self.fuel_Pr[i, j+1])
                #self.steam_Nu[i+1, j] = self.entryNuDh(self.steam_z[i+1], self.steam_dh, self.steam_Re[i+1, j], self.steam_Pr[i+1, j])
                if self.fuel_Re[i, j+1] < 2300: 
                    self.fuel_Nu[i, j+1] = self.entryNuDh(self.fuel_z[j+1], self.fuel_dh, self.fuel_Re[i, j+1], self.fuel_Pr[i, j+1], self.fuel_eps, self.fuel_sqrtA)
                else:
                    self.fuel_Nu[i, j+1] = 0.023*self.fuel_Re[i, j+1]**0.8*self.fuel_Pr[i, j+1]**0.3
                    
                if self.steam_Re[i+1, j] < 2300:
                    self.steam_Nu[i+1, j] = self.entryNuDh(self.steam_z[i+1], self.steam_dh, self.steam_Re[i+1, j], self.steam_Pr[i+1, j], self.steam_eps, self.steam_sqrtA)
                else:
                    self.steam_Nu[i+1, j] = 0.023*self.steam_Re[i+1, j]**0.8*self.steam_Pr[i+1, j]**0.4
                
                self.fuel_h[i, j+1] = self.fuel_Nu[i, j+1]*self.fuel_k[i, j+1]/self.fuel_dh
                self.steam_h[i+1, j] = self.steam_Nu[i+1, j]*self.steam_k[i+1, j]/self.steam_dh
                
                #update lplus and friction factors
                self.fuel_lplus[i, j+1] = self.lplus(self.fuel_z[j+1], self.fuel_dh, self.fuel_Re[i, j+1], self.fuel_sqrtA)
                self.steam_lplus[i+1, j] = self.lplus(self.steam_z[i+1], self.steam_dh, self.steam_Re[i+1, j], self.steam_sqrtA)
                self.fuel_f[i, j+1] = self.devflow_friction(self.fuel_lplus[i, j+1], self.fuel_eps, self.fuel_Re[i, j+1], self.fuel_dh, self.fuel_sqrtA)
                self.steam_f[i+1, j] = self.devflow_friction(self.steam_lplus[i+1, j], self.steam_eps, self.steam_Re[i+1, j], self.steam_dh, self.steam_sqrtA)
                        
        #output steam and fuel conditions
        self.fuel = [self.fuel_T[:, :], self.fuel_P[:, :]]
        self.steam = [self.steam_T[:, :], self.steam_P[:, :]]
        
        return(self.fuel, self.steam)
    
    
    def odeSolver(self, t, T):
        #take dimensions and initial conditions from elsewhere in the function        
        rows = int(self.dims[2])
        columns = int(self.dims[3])
        
        initial_fuel_Temps = T[0:int(rows*(columns+1))]
        initial_steam_Temps = T[int(rows*(columns+1)):]
        
        initial_fuel_T = initial_fuel_Temps.reshape(int(rows), int(columns)+1)
        initial_steam_T = initial_steam_Temps.reshape(int(rows)+1, int(columns))
        
        
        dTdt_fuel = np.zeros(shape = [rows, columns+1])
        dTdt_steam = np.zeros(shape = [rows+1, columns])
        
        for j in range(0, columns):
            for i in range(0, rows):
                
                #use steady-state model initial approximations for friction factors, properties, etc
                
                dTdt_fuel[i, j] = (-self.fuel_in[1]/self.dims[2]*(initial_fuel_T[i, j] - initial_fuel_T[i, j+1])/(self.fuel_cs*(self.fuel_z[1] + self.fuel_z[0])*self.fuel_rho[i, j]) - self.U[i, j]*self.hx_area*(initial_fuel_T[i, j] - initial_steam_T[i, j])/(self.fuel_cp[0,0]*(self.fuel_cs*(self.fuel_z[1] - self.fuel_z[0])*self.fuel_rho[i, j])))
                dTdt_steam[i, j] =  (-self.steam_in[1]/self.dims[3]*(initial_steam_T[i, j] - initial_steam_T[i+1, j])/(self.steam_cs*(self.steam_z[1] - self.steam_z[0])*self.steam_rho[i, j]) + self.U[i, j]*self.hx_area*(initial_fuel_T[i, j] - initial_steam_T[i, j])/(self.steam_cp[i, j]*(self.steam_cs*(self.steam_z[1] - self.steam_z[0])*self.steam_rho[i, j])))
                #dTdt_fuel[i, j] = (-self.fuel_in[1]/self.dims[2]*(initial_fuel_T[i, j] - initial_fuel_T[i, j+1])/(self.fuel_cs*(self.fuel_z[1] + self.fuel_z[0])*10) - 0.1/(1200*(self.fuel_cs*(self.fuel_z[1] - self.fuel_z[0])*10)))
                #dTdt_steam[i, j] =  (-self.steam_in[1]/self.dims[3]*(initial_steam_T[i, j] - initial_steam_T[i+1, j])/(self.steam_cs*(self.steam_z[1] - self.steam_z[0])*5) + 0.1/(1900*(self.steam_cs*(self.steam_z[1] - self.steam_z[0])*5)))
                
        T = np.concatenate([dTdt_fuel.ravel(), dTdt_steam.ravel()])
        return T
    
    
    def emptyarray(self, dim1, dim2, direction):
        #take number of channels and if the array should have an extra row or column
        #extra column is for fuel channel initialization, and extra row is for steam channel
        if direction == 1:
            narrow = 1
            wide = 0
        elif direction == 2:
            narrow = 0
            wide = 1
        empty = np.empty(shape = [dim1+1*wide, dim2+1*narrow], 
                          dtype = float)
        empty[:, :] = 0
        return(empty) 
        
    def reynolds(self, density, velocity, diameter, viscosity):
        #basic reynolds number calculation
        re = density*velocity*diameter/viscosity
        return(re)
    
    """        
    def entryNuDh(self, z, dh, Re, Pr):
        #Muzychka & Yovanovich, universal wall temperature nusselt number for combined entry region
        #there is a more in-depth correlation proposed in 10.1115/1.1643752#
        C4 = 1
        fPr = (0.564)/(1+(1.664*Pr**(1/6))**(9/2))**(2/9)
        if z == 0:
            zstar = 0.01 #initial assumption for this can break the heat exchanger?
        else:
            zstar = z/(dh*Re*Pr)
        Nu = C4*fPr/(zstar**0.5)
        return(Nu)
    """

    def entryNuDh(self, z, dh, Re, Pr, eps, sqrtA):
        C1 = 3.24
        C2 = 1
        C3 = 0.409
        C4 = 1
        
        ReRootA = Re*sqrtA/dh
        fRerootA = 12/(eps**0.5*(1+eps)*(1-192*eps/math.pi**5*math.tanh(math.pi/(2*eps))))
        
        if z == 0:
            zstar = 0.0001
        else:
            zstar = z/(dh*Re*Pr)
            
        Nu = (((C2*C3*(fRerootA/zstar)**(1/3))**5 + (C1*(fRerootA/(8*math.pi**0.5*eps**0.1)))**5))**(0.2)
        return(Nu)

    
    def lplus(self, L, Dh, ReDh, sqrtA):
        #Muzychka & Yovanovich
        lplus = L*Dh/(ReDh*sqrtA**2)
        return(lplus)
        
    def devflow_friction(self, lplus, eps, ReDh, Dh, sqrtA):
        #developing flow friction factor - give a minimum length to prevent divide by 0 errors
        if lplus == 0:
            lplus = 1
        f = (((3.44/(lplus**0.5))**2 + (12/((eps**0.5) * (1 + eps) * (1 - 192*eps/(math.pi**5)*np.tanh(math.pi/(2*eps)))))**2)*0.5)/(ReDh*sqrtA/Dh)
        return(f)
    
fuelin = [{'O2':5, 'CO2':92, 'H2O':3, 'CH4':0}, 0.0001, 1200, 1500000]
steamin = [{'H2O': 100}, 0.0001, 500, 1000000]
dimensions = [0.002, 0.001, 10, 20, 0.0002]

hx1 = crossflow_hx(fuelin, steamin, dimensions)
fuelout, steamout = hx1.solvehx()

#estimate initial temperature profile for the exchanger
fuelinitials = fuelin[2]*np.ones(shape = (dimensions[2], dimensions[3] + 1))
steaminitials = steamin[2]*np.ones(shape = (dimensions[2]+1, dimensions[3]))

odeTemps = np.concatenate([fuelinitials.ravel(), steaminitials.ravel()])

sol = solve_ivp(hx1.odeSolver, [0, 10], odeTemps, method = "RK45", t_eval = [0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10])
Tprofile = sol['y'][:, :]

#fuel_profile = Tprofile[:, 5][:int(dimensions[2]*(dimensions[3]+1))]
#steam_profile = Tprofile[:, 5][int(dimensions[2]*(dimensions[3]+1)):]

#fuel_profile = np.flip(fuel_profile.reshape(dimensions[2], dimensions[3] + 1))
#steam_profile = np.flip(steam_profile.reshape(dimensions[2] + 1, dimensions[3]), axis = 1)

#empty charts to hold average temperature profile data
plot_T_profile_fuel = np.zeros(shape = (9, 21))
plot_T_profile_steam = np.zeros(shape = (9, 11))

for i in range(0, 8):
    #grab time profile
    fuel_profile = Tprofile[:, i][:int(dimensions[2]*(dimensions[3]+1))]
    steam_profile = Tprofile[:, i][int(dimensions[2]*(dimensions[3]+1)):]
    
    #flip profile
    fuel_profile = np.flip(fuel_profile.reshape(dimensions[2], dimensions[3] + 1))
    steam_profile = np.flip(steam_profile.reshape(dimensions[2] + 1, dimensions[3]), axis = 0)

    for j in range(0, 20):
        plot_T_profile_fuel[i, j+1] = np.average(fuel_profile[:, j])
    for j in range(0, 10):
        plot_T_profile_steam[i, j+1] = np.average(steam_profile[j, :])
        
for i in range(0, 20):
    plot_T_profile_fuel[8, i+1] = np.average(fuelout[0][:, i])

for i in range(0, 10):
    plot_T_profile_steam[8, i+1] = np.average(steamout[0][i, :])
    
plot_T_profile_fuel[:, 0] = [0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, '999']
plot_T_profile_steam[:, 0] = [0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, '999']

np.savetxt('fuel_profiles.csv',plot_T_profile_fuel, delimiter=',', fmt='%10f')
np.savetxt('steam_profiles.csv',plot_T_profile_steam, delimiter=',', fmt='%10f')
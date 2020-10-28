#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:52:05 2020
Modified on 27 Oct

Cross flow heat exchanger model function

@author: afurlong
"""

def crossflow_hx(fuel_comp, fuel_flow, fuel_temp, fuel_p, 
                 steam_flow, steam_temp, steam_p,
                 fuel_height, fuel_width, steam_height, steam_width,
                 n_fuelch, n_steamch):
    #fuel_comp is cantera notation list
    #fuel_flow is molar flow rate, kg/s
    #fuel_temp is heat exchanger section inlet temperature in K
    #fuel_p is heat exchange section pressure in Pa
    #steam_flow is mass flow rate of steam, in kg/s
    #steam_temp is steam inlet temperature in K
    #steam_p is steam inlet pressure in Pa
    #fuel_height/fuel_width is fuel channel dimensions in m
    #steam_height/width is steam channel dimensions in m
    #n_fuelch is total number of fuel channels across
    #n_steamch is total number of steam channels across
    
    #define geometry of hx areas
    fuel_cs = fuel_height*fuel_width
    steam_cs = steam_width*steam_height
    fuel_dh = 2*fuel_height*fuel_width/(fuel_height+fuel_width)
    steam_dh = 2*steam_height*steam_width/(steam_height+steam_width)
    hx_area = steam_width*fuel_width
    
    #create arrays for temperatures of each channel, plus either a row or column for inlet and outlet
    fuel_temps = emptyarray(n_fuelch, n_steamch, 1)
    fuel_temps[:, 0] = fuel_temp
    fuel_temps[:, 1:] = 0
    
    steam_temps = emptyarray(n_fuelch, n_steamch, 2)
    steam_temps[0, :] = steam_temp
    steam_temps[1:, :] = 0
    
    #use cantera to get initial fluid densities
    fuel = ct.Solution('gri30.xml')
    fuel.transport_model = 'Mix'
    fuel.X = fuel_comp
    fuel.TP = fuel_temp, fuel_p
    fuel_density = emptyarray(n_fuelch, n_steamch, 1)
    fuel_density[:, 0] = fuel.density
    
    steam_density = emptyarray(n_fuelch, n_steamch, 2)
    steam = ct.Solution('gri30.xml')
    steam.transport_model = 'Mix'
    steam.X = {'H2O': 1}
    steam.TP = steam_temp, steam_p
    steam_density[0, :] = steam.density
    
    #set up array for fluid velocities, reynolds numbers, heat transfer coefficients
    fuel_velocity = emptyarray(n_fuelch, n_steamch, 1)
    fuel_velocity[:, 0] = fuel_flow/(fuel.density*fuel_cs*n_fuelch)
    
    steam_velocity = emptyarray(n_fuelch, n_steamch, 2)
    steam_velocity[0, :] = steam_flow/(steam.density*steam_cs*n_steamch)
    
    fuel_re = emptyarray(n_fuelch, n_steamch, 1)
    fuel_re[:, 0] = reynolds(fuel.density, fuel_velocity[0,0], fuel_dh, fuel.viscosity)
    
    steam_re = emptyarray(n_fuelch, n_steamch, 2)
    steam_re[0, :] = reynolds(steam.density, steam_velocity[0,0], steam_dh, steam.viscosity)
        
    fuel_viscosity = emptyarray(n_fuelch, n_steamch, 1)
    fuel_viscosity[:, 0] = fuel.viscosity
    steam_viscosity = emptyarray(n_fuelch, n_steamch, 2)
    steam_viscosity[0, :] = steam.viscosity
    
    fuel_k = emptyarray(n_fuelch, n_steamch, 1)
    fuel_k[:, 0] = fuel.thermal_conductivity
    steam_k = emptyarray(n_fuelch, n_steamch, 2)
    steam_k[0, :] = steam.thermal_conductivity
    fuel_cp = emptyarray(n_fuelch, n_steamch, 1)
    fuel_cp[:, 0] = fuel.cp_mass
    steam_cp = emptyarray(n_fuelch, n_steamch, 2)
    steam_cp[0, :] = steam.cp_mass
    
    fuel_Pr = emptyarray(n_fuelch, n_steamch, 1)
    fuel_Pr[:, 0] = fuel.viscosity*fuel.cp_mass/fuel.thermal_conductivity
    steam_Pr = emptyarray(n_fuelch, n_steamch, 2)
    steam_Pr[0, :] = steam.viscosity*steam.cp_mass/steam.thermal_conductivity
    
    fuel_nu = emptyarray(n_fuelch, n_steamch, 1)
    fuel_nu[:, 0] = 0.023*(fuel_re[0, 0]**0.8)*(fuel_Pr[0, 0])**(0.3)
    steam_nu = emptyarray(n_fuelch, n_steamch, 2)
    steam_nu[0, :] = 0.023*(steam_re[0, 0]**0.8)*(steam_Pr[0, 0])**(0.4)
    
    fuel_h = emptyarray(n_fuelch, n_steamch, 1)
    fuel_h[:, 0] = fuel_nu[0, 0]*fuel.thermal_conductivity/fuel_dh
    
    steam_h = emptyarray(n_fuelch, n_steamch, 2)
    steam_h[0, :] = steam_nu[0, 0]*steam.thermal_conductivity/steam_dh
    
    U = np.empty(shape = [n_fuelch, n_steamch], dtype = float)
    U[:, :] = 0
    Q = np.empty(shape = [n_fuelch, n_steamch], dtype = float)
    Q[:, :] = 0
   
    #iterative steps to solve for all temperatures after setup
    for j in range(0, n_steamch):
        for i in range(0, n_fuelch):
            #use steel at 45 W/mK, 1 mm thick
            U[i, j] = 1/(1/fuel_h[i, j] + 1/steam_h[i, j] + 0.001/45)
            Q[i, j] = U[i, j]*hx_area*(fuel_temps[i, j] - steam_temps[i, j])
            fuel_temps[i, j+1] = (fuel_density[i, j]*fuel_velocity[i, j]*fuel_cs*fuel_cp[i, j]*fuel_temps[i, j] - Q[i, j])/(fuel_density[i, j]*fuel_velocity[i, j]*fuel_cp[i, j]*fuel_cs)
            steam_temps[i+1, j] = (steam_density[i, j]*steam_velocity[i, j]*steam_cs*steam_cp[i, j]*steam_temps[i, j] + Q[i, j])/(steam_density[i, j]*steam_velocity[i, j]*steam_cp[i, j]*steam_cs)
            #update mixture for a new set of fluid properties, and update arrays as relevant
            fuel.TP = fuel_temps[i, j+1], fuel_p
            steam.TP = steam_temps[i+1, j], steam_p

            fuel_density[i, j+1] = fuel.density
            steam_density[i+1, j] = steam.density
            fuel_velocity[i, j+1] = fuel_velocity[i, j]*fuel_density[i, j+1]/fuel_density[i, j]
            steam_velocity[i+1, j] = steam_velocity[i, j]*steam_density[i+1, j]/steam_density[i, j]
            fuel_viscosity[i, j+1] = fuel.viscosity
            steam_viscosity[i+1, j] = steam.viscosity
            fuel_k[i, j+1] = fuel.thermal_conductivity
            steam_k[i+1, j] = steam.thermal_conductivity
            fuel_cp[i, j+1] = fuel.cp_mass
            steam_cp[i+1, j] = steam.cp_mass
            fuel_Pr[i, j+1] = fuel_viscosity[i, j+1]*fuel_cp[i, j+1]/fuel_k[i, j+1]
            steam_Pr[i+1, j] = steam_viscosity[i+1, j]*steam_cp[i+1, j]/steam_k[i+1, j]
            
            fuel_re[i, j+1] = reynolds(fuel_density[i, j+1], fuel_velocity[i, j+1], fuel_dh, fuel_viscosity[i, j+1])
            steam_re[i+1, j] = reynolds(steam_density[i+1, j], steam_velocity[i+1, j], steam_dh, steam_viscosity[i+1, j])
            
            #calculate new nusselt numbers and heat transfer coefficients
            fuel_nu[i, j+1] = 0.023*(fuel_re[i, j+1]**0.8)*(fuel_Pr[i, j+1])**(0.3)
            steam_nu[i+1, j] = 0.023*(steam_re[i+1, j]**0.8)*(steam_Pr[i+1, j])**(0.4)
            fuel_h[i, j+1] = fuel_nu[i, j+1]*fuel_k[i, j+1]/fuel_dh
            steam_h[i+1, j] = steam_nu[i+1, j]*steam_k[i+1, j]/steam_dh
            
    return(fuel_temps, steam_temps)
    
def reynolds(density, velocity, diameter, viscosity):
    re = density*velocity*diameter/viscosity
    return(re)
    
def emptyarray(column, row, dim):
    if dim == 1:
        narrow = 1
        wide = 0
    elif dim == 2:
        narrow = 0
        wide = 1
    empty = np.empty(shape = [column+1*wide, row+1*narrow], dtype = float)
    empty[:, :] = 0
    return(empty)
    

####main
import numpy as np
import cantera as ct
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

output = crossflow_hx({'O2':5, 'CO2': 92, 'H2O': 3, 'CH4':0}, 0.3, 1100, 1500000, 
                      1, 373, 100000, 
                      0.001, 0.001, 0.001, 0.002, 
                      1000, 2000)

vmin = min(np.min(output[0]), np.min(output[1]))
vmax = max(np.max(output[0]), np.max(output[1]))

fig, (fuelplot, steamplot) = plt.subplots(1, 2, sharey = True, sharex = True)

fuelplot = sns.heatmap(output[0], cmap = cmap.hot, cbar = False, 
                       ax = fuelplot, square = True, 
                       xticklabels = 200, yticklabels = 200,
                       vmin = vmin, vmax = vmax)

steamplot = sns.heatmap(output[1], cmap = cmap.hot, cbar = False, 
                        ax = steamplot, square = True, 
                        xticklabels = 200, yticklabels = 200,
                        )
plt.show()


#fig, axs = plt.subplots(2)

#fuel = plt.imshow(output[0], cmap = cmap.hot)
#steam = plt.imshow(output[1], cmap = cmap.hot)

#axs[0].plot(fuel)
#axs[1].plot(steam)

#plt.plot()
#plt.clim(650, 800)
#plt.colorbar()
#plt.show()

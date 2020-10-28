# cat-hx-microreactor
Catalystic Heat Exchange Microreactor for Oxy-Fuel Combustion Reduced Order Modelling
Developed in Python with Cantera for reactions and fluid properties
Main components consist of cross-flow heat exchanger modelling in micro-channels and catalyzed reactions in packed beds

Cross-flow heat exchanger model:
Shows fuel flow in the x-direction, utility flow in the y-direction, and heat transfer in the z-direction. 
Fuel transfers heat in the y-direction to other fuel but the effect is negligible
Steam transfers heat in the x-direction to other fuel but the effect is negligible
Natural convection has no impact in each individual cross-flow channel, and there is no additional heat transfer beyond the bulk motion of fluid in the direction of fluid flow
Channel walls are assumed to be infinitely thin, with no heat transfer occurring

Packed bed reactor model:

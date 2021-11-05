#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 22:45:56 2021

@author: christian Computational Physics : N body problem solver class with Runge Kutta 4
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['axes.titlesize'] = 28 # change the size of the title
plt.rcParams['axes.labelsize'] = 20 # change la taille du label des axes seulement
plt.rcParams['axes.linewidth'] = 1 # augmente l'épaisseur du contour de la figure
plt.rcParams['lines.linewidth'] = 2 # augmente l'épaisseur des lignes
plt.rcParams['xtick.labelsize'] = 17 # augmente la taille des nombres gradués en x
plt.rcParams['ytick.labelsize'] = 17 # pareil mais en y

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"]})

#%%

# =============================================================================
# Definition of class and functions
# =============================================================================

def F(i,j,k,l,u,epsilon = 0):
    """ u must be an array or a list"""
    return (u[i]-u[j])/((u[i]-u[j])**2+(u[k]-u[l])**2+epsilon)**1.5

class N_body_Problem(object):
    
    def __init__(self,N,masses,initial_conditions,normalisation_constant = 1):
        """ Initiate the object Problem with N the number of bodies the masses as a list or array of the masses (ordered)
        and the initial conditions (also ordered in a partcular way"""
        self.N = N
        self.cst = normalisation_constant
        self.masses = masses
        self.initial_conditions = initial_conditions
    
    def ODEs_system(self,u):
        """ Create a vectorial function which respresents the N*4 equations of the problem in order to be solved by Runge-Kutta 4
        method"""
        f = []
        for i in range(self.N):
            f2 = 0 ; f3 = 0
            f.append(u[1+4*i])
            for j in range(self.N):
                if j == i:
                    f2 += 0
                else:
                    f2 += self.cst*self.masses[j]*F(4*j,4*i,4*j+2,4*i+2,u)
            f.append(f2)
            f.append(u[3+4*i])
            for j in range(self.N):
                if j == i:
                    f3 += 0
                else:
                    f3 += self.cst*self.masses[j]*F(4*j+2,4*i+2,4*j,4*i,u)
            f.append(f3)
        return np.array(f)
    
    def RK4_Solver(self,t_0 = 0, t_f = 100, h = 0.1):
        """ Solve the ODEs_system of the N body problem according to the initial conditions using RK4 method """
        self.M = int((t_f-t_0)/h)
        time = np.linspace(t_0,t_f,self.M)
        u_n = self.initial_conditions
        results = []
        for i in range(self.M):
            k1 = h*self.ODEs_system(u_n)
            k2 = h*self.ODEs_system(u_n + 0.5*k1)
            k3 = h*self.ODEs_system(u_n + 0.5*k2)
            k4 = h*self.ODEs_system(u_n+k3)
            u_n = u_n + (k1 + 2*k2 + 2*k3 + k4)/6
            results.append(u_n)
        self.time,self.results = np.array(time), np.array(results)
        
    def Trajectories(self,colors,radius,axes_limit,interval = 20, initial_time = 0, final_time = 100, h = 0.1,text_ypos = - 1,title=''):
        """ Animate the trajectories of the N body problem usingthe solutions of the RK4 method"""
        plt.style.use('dark_background')
        self.RK4_Solver(t_0 = initial_time, t_f = final_time, h = h)
        self.figure,self.axes = plt.subplots(figsize = (9,9))
        self.axes_limit = axes_limit
        self.axes.set_title(title)
        #plt.axis('off')
        self.axes.set_xlabel(r'Distance normalisée avec $d_{Soleil-Uranus} \simeq 17$ UA')
        self.axes.set_aspect('equal')
        self.trajectories_data = []
        self.lines = tuple()
        self.balls = tuple()
        self.axes.text(1.2,-1.75,r'$\tau$ affichée en année terrestre',horizontalalignment='center')
        self.time_text = self.axes.text(1.5,text_ypos,'',horizontalalignment='center',bbox=dict(facecolor='none',edgecolor = 'white',pad = 10.0))
        for i in range(self.N):
            self.lines += tuple(plt.plot([],[],color=colors[i],lw = 2))
            self.balls += tuple([plt.Circle((self.initial_conditions[4*i],self.initial_conditions[4*i+2]),radius[i],color = colors[i])])
            self.axes.add_patch(self.balls[i])
            for k in range(2):
                self.trajectories_data.append([])
        
        def init():
            """ Function to initiate the animation"""
            self.time_text.set_text('0')
            self.axes.set_xlim(-self.axes_limit,self.axes_limit)
            self.axes.set_ylim(-self.axes_limit,self.axes_limit)
            for i in self.lines:
                i.set_data([],[])
            return self.lines + self.balls + tuple([self.time_text])
        
        def update(frames):
            """ Function to update the animation"""
            if frames == int(self.time.size-1):
                self.trajectories_data = []
                for i in range(2*self.N):
                    self.trajectories_data.append([])              
            else:
                for i in range(self.N):
                    self.balls[i].set_center((self.results[frames,4*i],self.results[frames,4*i+2]))
                for i in range(2*self.N):
                    self.trajectories_data[i].append(self.results[frames,2*i])
                for i in range(self.N):
                    self.lines[i].set_data(self.trajectories_data[2*i],self.trajectories_data[2*i+1])
            self.time_text.set_text(r'$\tau = {}$'.format(round(frames*h*11.457,2)))
            return self.lines + self.balls + tuple([self.time_text])
        
        self.interval = interval
        self.animation = animation.FuncAnimation(self.figure,update,frames = int(self.time.size), init_func = init,
                                                 interval = self.interval, blit = False, repeat = True)
    def Save_animation(self,title):
        """ Save the animation of the solutions """
        self.animation.save(title, writer=animation.PillowWriter(fps=30) )
        
#%%
# =============================================================================
# Epsilon test 
# =============================================================================
masses_eps = np.array([1,1,1])
CI_eps = 2*np.array([-4,0,0,0,4,0,0,0,1,0,5,0])
eps_P = N_body_Problem(N = 3, masses = masses_eps, initial_conditions = CI_eps)
eps_P.Trajectories(colors = ['white','orange','white'], radius = [0.5,0.5,0.5], axes_limit = 12,final_time=400,h = 1,text_ypos=-6
                             ,title = r'Solution $\epsilon$ non physique')
plt.close()
#%%

eps_P.Save_animation('epsilon.gif')

#%%
# =============================================================================
# Euler coplanar solutions 0
# =============================================================================
masses_Euler_0 = np.array([1,1,1])
coefs_0 = [-(masses_Euler_0[1]+masses_Euler_0[2]),-(2*masses_Euler_0[1]+3*masses_Euler_0[2]),-(masses_Euler_0[1]+3*masses_Euler_0[2]),
         (3*masses_Euler_0[0]+masses_Euler_0[1]),(3*masses_Euler_0[0]+2*masses_Euler_0[1]),(masses_Euler_0[0]+masses_Euler_0[1])]
poly_0 = np.polynomial.polynomial.Polynomial(coefs_0,domain = [-1,1])
roots_0 = poly_0.roots()
CI_Euler_0 = np.array([-2,0,0,-1,0,0,0,0,2,0,0,1])
Euler_Problem_0 = N_body_Problem(N = 3, masses = masses_Euler_0, initial_conditions = CI_Euler_0)
Euler_Problem_0.Trajectories(colors = ['white','orange','white'], radius = [0.5,0.5,0.5], axes_limit = 20,final_time=300,h = 0.5,text_ypos=-6
                             ,title = 'Exemple trivial d\'une solution d\'Euler')
#plt.close()
#%%

Euler_Problem_0.Save_animation('Euler_0.gif')

#%%
# =============================================================================
# Euler coplanar solutions 1
# =============================================================================
masses_Euler_1 = np.array([1,2,3])
coefs_1 = [-(masses_Euler_1[1]+masses_Euler_1[2]),-(2*masses_Euler_1[1]+3*masses_Euler_1[2]),-(masses_Euler_1[1]+3*masses_Euler_1[2]),
         (3*masses_Euler_1[0]+masses_Euler_1[1]),(3*masses_Euler_1[0]+2*masses_Euler_1[1]),(masses_Euler_1[0]+masses_Euler_1[1])]
poly_1 = np.polynomial.polynomial.Polynomial(coefs_1,domain = [-1,1])
roots_1 = poly_1.roots() ; d = roots_1[-1].real

CI_Euler_1 = np.array([-8,0,0,-(d+5/3)/(d+1/3),-6,0,0,-(d-1/3)/(d+1/3),2*d-6,0,0,1])
Euler_Problem_1 = N_body_Problem(N = 3, masses = masses_Euler_1, initial_conditions = CI_Euler_1)
Euler_Problem_1.Trajectories(colors = ['white','white','white'], radius = [0.7,0.7,0.7], axes_limit = 17.4,final_time=340,h = 0.5,text_ypos=-14,
                             title='Exemple d\'une solution d\'Euler perturbée')
plt.close()
#%%

Euler_Problem_1.Save_animation('Euler_1_chaos.gif')

#%%
# =============================================================================
# Figure 8 Solutions
# =============================================================================
masses_F8 = [1,1,1]
x1 = 0.97000436 ; y1 = -0.24308753 ; x_dot_3 = -0.93240737 ; y_dot_3 = -0.86473146
CI_F8 = [x1,-0.5*x_dot_3,y1,-0.5*y_dot_3,-x1,-0.5*x_dot_3,-y1,-0.5*y_dot_3,0,x_dot_3,0,y_dot_3]
F8_Problem = N_body_Problem(N = 3, masses = masses_F8, initial_conditions = CI_F8)
F8_Problem.Trajectories(colors = ['white','white','white'], radius = [0.06,0.06,0.06], axes_limit = 1.2,final_time=19,h = 0.025,text_ypos=-0.7,
                        title='Solution dite figure 8 ou infini')
plt.close()
#%%

F8_Problem.Save_animation('F8.gif')

#%%
# =============================================================================
# Lagrange equilateral triangle solutions refaire chaos
# =============================================================================
masses_L = [1,1,1]

CI_L = [0-0.5+0.0001,1.3*np.cos(np.pi/3),-1/3,-1.3*np.sin(np.pi/3),1-0.5,1.3*np.cos(np.pi/3),0-1/3,1.3*np.sin(np.pi/3),0.5-0.5,-1.3,np.sin(np.pi/3)-1/3,0]
L_Problem = N_body_Problem(N = 3, masses = masses_L, initial_conditions = CI_L)
L_Problem.Trajectories(colors = ['white','white','white'], radius = [0.2,0.2,0.2], axes_limit = 4,final_time=70,h = 0.15,text_ypos=-1.7,
                       title ='Exemple d\'une solution dite de Lagrange perturbée')
plt.close()
#%%

L_Problem.Save_animation('Lagrange_chaos.gif')

#%%
# =============================================================================
# 3 Rings
# =============================================================================
masses_3R = [1,1,1]

CI_3R = [-0.0347,0.2495,1.1856,-0.1076,0.2693,0.2059,-1.0020,-0.9396,-0.2328,-0.4553,-0.5978,1.0471]
CI_3R1 = [1.1856,-0.1076,0.0347,-0.2495,-1.0020,-0.9396,-0.2693,-0.2059,-0.5978,1.0471,0.2328,0.4553]
R_Problem = N_body_Problem(N = 3, masses = masses_3R, initial_conditions = CI_3R1)
R_Problem.Trajectories(colors = ['orange','orange','white'], radius = [0.06,0.06,0.06], axes_limit = 1.6,final_time=8.7,h = 0.01,text_ypos=-1,
                       title='Solution particulière "trois anneaux"')
plt.close()
#%%

R_Problem.Save_animation('3_rings.gif')

#%%
# =============================================================================
# Flower in a circle
# =============================================================================
masses_FC = [1,1,1]

CI_FC = [-0.602885898116520+0.24,0.122913546623784,1.059162128863347-1,0.747443868604908,0.252709795391000+0.24,-0.019325586404545,
         1.058254872224370-1,1.369241993562101,-0.355389016941814+0.24,-0.103587960218793,1.038323764315145-1,-2.116685862168820]
FC_Problem = N_body_Problem(N = 3, masses = masses_FC, initial_conditions = CI_FC)
FC_Problem.Trajectories(colors = ['white','white','orange'], radius = [0.04,0.04,0.04], axes_limit = 1,final_time=6.75,h = 0.01,text_ypos=-1,
                        title='Solution avec une "fleur dans un cercle"')
plt.close()
#%%

FC_Problem.Save_animation('Flower.gif')

#%%
# =============================================================================
# Two Ovals à refaire
# =============================================================================
masses_OvF = [1,1,1]

CI_OvF = [0.486657678894505,-0.182709864466916,0.755041888583519,0.363013287999004,-0.681737994414464,-0.579074922540872,0.293660233197210
          ,-0.748157481446087,-0.022596327468640,0.761784787007641,-0.612645601255358,0.385144193447218]
OvF_Problem = N_body_Problem(N = 3, masses = masses_OvF, initial_conditions = CI_OvF)
OvF_Problem.Trajectories(colors = ['orange','white','white'], radius = [0.06,0.06,0.06], axes_limit = 1.5,final_time=12,h = 0.02,text_ypos=-1,
                         title='Solution avec deux ovales')
plt.close()
#%%

OvF_Problem.Save_animation('2_ovales.gif')

#%%
# =============================================================================
# Circle bis
# =============================================================================
masses_Cb = [1,1,1]

CI_Cb = [1.666163752077218-1-0.187,0.841202975403070,-1.081921852656887+1,0.029746212757039,0.974807336315507-1-0.187,0.142642469612081
          ,-0.545551424117481+1,-0.492315648524683,0.896986706257760-1-0.187,-0.983845445011510,-1.765806200083609+1,0.462569435774018]
Cb_Problem = N_body_Problem(N = 3, masses = masses_Cb, initial_conditions = CI_Cb)
Cb_Problem.Trajectories(colors = ['orange','white','white'], radius = [0.04,0.04,0.04], axes_limit = 1.1,final_time=11.4,h = 0.025,text_ypos=-1,
                        title='Solution particulière "fleur dans un cercle" bis')
plt.close()
#%%

Cb_Problem.Save_animation('Circle_bis.gif')

#%%
# =============================================================================
# N = 4 Lagrange refaire 
# =============================================================================
masses_L4 = [1,1,1,1]
CI_L4 = [-1,np.cos(np.pi/4),-1,-np.sin(np.pi/4),1,np.cos(np.pi/4),-1,np.sin(np.pi/4),1,-np.cos(np.pi/4),1,np.sin(np.pi/4),-1,-np.cos(np.pi/4),1,
         -np.sin(np.pi/4)]
P_L4 = N_body_Problem(N = 4, masses = masses_L4, initial_conditions = CI_L4)
P_L4.Trajectories(colors = ['white','white','white','white'], radius = [0.15,0.15,0.15,0.15], axes_limit = 4,final_time=28.5*3,h = 0.2,text_ypos=0,
                  title='Solution de Lagrange')
plt.close()
#%%

P_L4.Save_animation('Lagrange_4.gif')

#%%
# =============================================================================
# Figure 8 + one small planet
# =============================================================================
masses_F8_bis = [1,1,1,1e-3]
x1 = 0.97000436 ; y1 = -0.24308753 ; x_dot_3 = -0.93240737 ; y_dot_3 = -0.86473146
CI_F8_bis = [-x1,-0.5*x_dot_3,-y1,-0.5*y_dot_3,x1,-0.5*x_dot_3,y1,-0.5*y_dot_3,0,x_dot_3,0,y_dot_3,-0.33,0,-0.3,0]
F8_Problem_bis = N_body_Problem(N = 4, masses = masses_F8_bis, initial_conditions = CI_F8_bis)
F8_Problem_bis.Trajectories(colors = ['yellow','white','orange','red'], radius = [0.1,0.1,0.1,0.05], axes_limit = 2,final_time=5,
                            h = 0.01,text_ypos=-0.5)
#%%        
masses_test = [1,1,1,1]
CI_test = [0.5,0,0,-1.5,-0.5,0,0,1.5,1.5,0,0,0.5,-1.5,0,0,-0.5]
test = N_body_Problem(N=4, masses = masses_test, initial_conditions = CI_test)
test.Trajectories(colors=['yellow','white','blue','red'], radius = [0.05,0.05,0.05,0.05], axes_limit = 1.61,final_time=25,
                            h = 0.1,text_ypos=0,title='Solution particulière pour N = 4')
plt.close()
#%%
test.Save_animation('N_4.gif')
#%%
# =============================================================================
# Sun - Earth - Moon dynamic
# =============================================================================
G = 6.674e-11 # m³/kg.s²
r_S = 696.342e6 # m radius of the sun
r_E = 6.625e6 # m radius of the Earth
r_M = 1.710e6 # m radius of the Moon
d_SE = 149.59e9 # m Sun-Earth distance
d_EM = 384.4e6 # m Eart-Moon distance
masses_SE = np.array([1.989e30,5.972e24,7.66e22]) # masses of Sun & Earth & Moon in kg
d0_SE = d_SE # normalisation of the distance
t0_SE = np.sqrt(d0_SE**3/(G*masses_SE[0]))
v_E = 29.78e3*(t0_SE/d0_SE) # mean speed of the Earth around the sun
v_M = 1.02e3*(t0_SE/d0_SE) # mean speed of the Moon around the Earth
cst_SE = t0_SE**2*G/d0_SE**3

#%%

IC_SE = np.array([0,0,0,0,d_SE/d0_SE,0,0,v_E,(d_SE+d_EM)/d0_SE,0,0,v_E+v_M])
SE = N_body_Problem(N = 3, masses = masses_SE, initial_conditions = IC_SE, normalisation_constant = cst_SE)
SE.Trajectories(colors = ['orange','blue','grey'], radius = np.array([r_S/d0_SE,1000*r_E/d0_SE,r_M/d0_SE]), axes_limit = 1.1,final_time=10,h = 0.01,text_ypos=-0.5)

#%%

SE.Save_animation('SE.gif')  
     
#%%
# =============================================================================
# Sun - Earth - Moon dynamic
# =============================================================================
G = 6.674e-11 # m³/kg.s²

r_S = 696.342e6 # m radius of the sun
r_E = 6.625e6 # m radius of the Earth
r_M = 1.710e6 # m radius of the Moon
r_Mars = (6.794e6)/2  # m radius of Mars
r_Mercury = (4.878e6)/2  # m radius of Mercury
r_V = (12.104e6)/2  # m radius of Venus
r_J = (142.984e6)/2  # m radius of Jupiter
r_Saturn = (120.536e6)/2  # m radius of Saturn
r_U = (51.118e6)/2  # m radius of Uranus
r_N = (49.528e6)/2  # m radius of Neptune



d_MercS = 57.8e9  # m Mercury-Sun mean distance
d_VS = 108.2e9  # m Venus-Sun mean distance
d_SE = 149.59e9  # m Sun-Earth distance
d_MarsS = 227.9e9  # m Mars-Sun mean distance
d_JS = 778.6e9 # m Jupiter-Sun mean distance
d_SS = 1433.5e9  # m Saturn-Sun mean distance
d_US = 2587.9e9 # m Uranus-Sun mean distance
d_NS = 4303.9e9 # m Neptune-Sun mean distance


M_S = 1.989e30 # kg mass of the Sun
M_T = 5.972e24 # kg mass of the Earth
M_Mars = 0.642e24 # kg mass of Mars
M_Merc = 0.33e24 # kg mass of Mercury
M_M = 7.66e22  # kg mass of the Moon
M_V = 4.87e24   # kg mass of Venus
M_J = 1899e24   # kg mass of Jupiter
M_Sat = 568e24   # kg mass of Saturn
M_U = 86.8e24   # kg mass of Uranus
M_N = 102e24   # kg mass of Neptun


#%%
masses_SE = np.array([M_S, M_Merc,  M_V, M_T, M_Mars]) # masses of Sun & planets
d0_SE = d_SE # normalisation of the distance
t0_SE = np.sqrt(d0_SE**3/(G*masses_SE[0]))

v_E = 29.78e3*(t0_SE/d0_SE) # mean speed of the Earth around the sun
v_Mars = 24.11e3*(t0_SE/d0_SE) # mean speed of Mars around the sun
v_Merc = 47.78e3*(t0_SE/d0_SE) # mean speed of Mercury around the sun
v_M = 1.02e3*(t0_SE/d0_SE) # mean speed of the Moon around the Earth
v_V = 35.02e3*(t0_SE/d0_SE) # mean speed of Venus around the Earth


cst_SE = t0_SE**2*G/d0_SE**3

#(d_SE+d_EM)/d0_SE,0,0,v_E+v_M
#%%

IC_SE = np.array([0,0,0,0,  d_MercS/d0_SE,0,0,v_Merc,   d_VS/d0_SE,0,0,v_V,  d_SE/d0_SE,0,0,v_E,    d_MarsS/d0_SE,0,0,v_Mars])
SE = N_body_Problem(N = 5, masses = masses_SE, initial_conditions = IC_SE, normalisation_constant = cst_SE)
SE.Trajectories(colors = ['orange', 'green', 'red', 'blue', 'brown'], radius = np.array([20*r_S/d0_SE, 1000*r_Mercury/d0_SE, 1000*r_E/d0_SE,  1000*r_V/d0_SE,  1000*r_Mars/d0_SE]), axes_limit = 1.8, final_time=13, h = 0.03,text_ypos=-1.5,
                title='Simulation du système solaire \n pour les quatres premières planètes')
plt.close()
#%%
SE.Save_animation('SMEMV.gif')     

  
#%%
masses_SE2 = np.array([M_S, M_J, M_Sat, M_U, M_N]) # masses of Sun & Earth & Moon in kg
d0_SE2 = d_US # normalisation of the distance
t0_SE2 = np.sqrt(d0_SE2**3/(G*masses_SE2[0]))

v_J = 13.07e3*(t0_SE2/d0_SE2) # mean speed of Jupiter around the Earth
v_S = 9.66e3*(t0_SE2/d0_SE2) # mean speed of Saturn around the Earth
v_U = 6.81e3*(t0_SE2/d0_SE2) # mean speed of Uranus around the Earth
v_N = 5.43e3*(t0_SE2/d0_SE2) # mean speed of Neptun around the Earth      
        
        
cst_SE2 = t0_SE2**2*G/d0_SE2**3
        
IC_SE = np.array([0,0,0,0,    d_JS/d0_SE2,0,0,v_J,   d_SS/d0_SE2,0,0,v_S,   d_US/d0_SE2,0,0,v_U,    d_NS/d0_SE2,0,0,v_N])
SE = N_body_Problem(N = 5, masses = masses_SE2, initial_conditions = IC_SE, normalisation_constant = cst_SE2)
a = 1300
SE.Trajectories(colors = ['orange', 'brown', 'grey', 'blue', 'red'], radius = np.array([500*r_S/d0_SE2,   a*r_J/d0_SE2, a*r_Saturn/d0_SE2,    a*r_U/d0_SE2,  a*r_N/d0_SE2]), axes_limit = 2, final_time=13,h = 0.03,text_ypos=-1.5
                ,title='Système solaire avec \n Jupiter, Saturne, Uranus et Neptune')
plt.close()
#%%
SE.Save_animation('JSUN.gif')    
        

        
        
        
        
        
        
        
        
        
        
        
        
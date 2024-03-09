#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Tue Dec  5 09:10:06 2023

# @author: mjelline
# """

# Stommel/ Taylor model modified from Marotzke, 2000
# Modified for nonlinear effect of ice lid on q and linear effect on Fs; 1/2024
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp

# font 
plt.rcParams['font.family'] = 'Helvetica Neue'


########################################################################


# Basic Stommel after Marotzke, 2000
# def q(DeltaT,DeltaS):
#     ########################################################################
#     #Original  MOC transport in m^3/sec as function of temperature and salinity
#     # difference between the boxes
#     flow=k*(-alpha*DeltaT+beta*DeltaS);
#     return flow

def Lid_gain_lookup(Lid):
    # Nonlinear effect of ice lid on deep salt-driven convection stirring speed parameterized.
    # Function returns a gain to be applied to transport coefficient
    Lid_min = 0.1
    Lid_crit = 0.6
    flow_0 = 1
    flow_min = flow_0 * 0.01
    flow_max = flow_0 * 1.5
    Lidrange = np.arange(0, 1, 0.01)
    Lid_gain = np.ones(len(Lidrange))
    n = 1.5

    if Lid <= Lid_min:
        Lid_gain[Lidrange <= Lid_min] = 1
    elif Lid_min < Lid < Lid_crit:
        Lid_gain[(Lid_crit > Lidrange) & (Lidrange > Lid_min)] = \
            1 + (flow_max - flow_min) * \
            ((Lidrange[(Lid_crit > Lidrange) & (Lidrange > Lid_min)] - Lid_min) ** n \
             / (Lid_crit - Lid_min) ** n)
    else:
        Lid_gain[:] = flow_min

    Lidpower = np.interp(Lid, Lidrange, Lid_gain)
    return Lidpower


########################################################################
def q(DeltaT, DeltaS, Lid):
    #     ########################################################################
    # Modified for ice lid effects: MOC transport in m^3/sec as function of temperature and salinity
    # difference between the boxes
    # Let lid extent L~sin(latitidue)
    return k * (-alpha * DeltaT + beta * DeltaS) * Lid_gain_lookup(Lid)


########################################################################
def steady_states(Fs, X):  # To do: adjust for constant lid extent effects
    ########################################################################
    # 3 steady solutions for Y:
    y1 = X / 2 - 0.5 * np.sqrt(X ** 2 + 4 * beta * Fs / k)
    if X ** 2 - 4 * beta * Fs / k > 0:
        y2 = X / 2 + 0.5 * np.sqrt(X ** 2 - 4 * beta * Fs / k)
        y3 = X / 2 - 0.5 * np.sqrt(X ** 2 - 4 * beta * Fs / k)
    else:
        y2 = np.nan
        y3 = np.nan
    Y = np.array([y1, y2, y3])
    return Y


########################################################################
def Lid_extent(time, time_max, hysteresis, Lid):
    T_period = time_max
    # T_period = 10^5*year
    Lid_max = 1
    Lid_0 = 0.1
    if hysteresis:
        return Lid_0 + np.sin(1 * np.pi * time / T_period)  # periodic growth and retreat
        # return Lid_0 + (Lid_max-Lid_0)*(time/T_period) #monotonic growth
    else:
        return Lid


########################################################################
def Fs_func(time, time_max, hysteresis, Lid):
    ########################################################################
    # total surface salt flux into northern box

    ########################################################################
    # Specify maximum and minimum of freshwater forcing range during the
    # run in m/year:
    FW_min = 0.1;
    # FW_max=5;
    ########################################################################

    if hysteresis:
        flux = FW_min * Lid_extent(time, time_max, hysteresis, Lid)

    else:
        flux = 2 * Lid_extent(time, time_max, hysteresis, Lid);

    # convert to total salt flux:
    return flux * area * S0 / year


########################################################################
def steady_states_L(Fs, X, Lid):
    ########################################################################
    # X=alpha*deltaT; Y = beta*deltaS
    # Vary Fs, let K vary with Lid = boxwidth / box depth, as per JL09, JFM
    # 3 steady solutions for Y:
    y1 = X / 2 - 0.5 * np.sqrt(X ** 2 + 4 * beta * (Fs) / (k * Lid_gain_lookup(Lid)))
    if X ** 2 - 4 * beta * Fs / (k * Lid_gain_lookup(Lid)) > 0:
        y2 = X / 2 + 0.5 * np.sqrt(X ** 2 - 4 * beta * (Fs) / (k * Lid_gain_lookup(Lid)))
        y3 = X / 2 - 0.5 * np.sqrt(X ** 2 - 4 * beta * (Fs) / (k * Lid_gain_lookup(Lid)))
    else:
        y2 = np.nan
        y3 = np.nan
    Y = np.array([y1, y2, y3])
    return Y


########################################################################
def steady_states_L_Fs_L(Fs, X, Lid):
    ########################################################################
    # Let Fs vary with Lid = boxwidth / box depth
    # 3 steady solutions for Y:
    y1 = X / 2 - 0.5 * np.sqrt(X ** 2 + 4 * beta * (Fs * (1 - Lid)) / (k * Lid_gain_lookup(Lid)))
    if X ** 2 - 4 * beta * Fs / (k * Lid_gain_lookup(Lid)) > 0:
        y2 = X / 2 + 0.5 * np.sqrt(X ** 2 - 4 * beta * (Fs * (1 - Lid)) / (k * Lid_gain_lookup(Lid)))
        y3 = X / 2 - 0.5 * np.sqrt(X ** 2 - 4 * beta * (Fs * (1 - Lid)) / (k * Lid_gain_lookup(Lid)))
    else:
        y2 = np.nan
        y3 = np.nan
    Y = np.array([y1, y2, y3])
    return Y


##RHS: For hsyteresis calcs-- recast time in terms of L: still to implement

# ########################################################################
# #New system
# def rhs_S_L(time,DeltaS,time_max,DeltaT,hysteresis):
#     ########################################################################
#     # right hand side of the equation for d/dt(DeltaS) and d/dt(L):
#     # Input: q in m^3/sec; FW is total salt flux into box
#     rhs=np.array([-(2*np.abs(q(DeltaT,DeltaS,Lid))*DeltaS+2*Fs_func(time,time_max,hysteresis,Lid))])
#     return rhs/V


########################################################################
## Main program:
########################################################################

# set parameters:
meter = 1;
year = 365 * 24 * 3600;
S0 = 35.0;
DeltaT = -10;
time_max = 100000 * year;
alpha = 0.2;  # (kg/m^3) K^-1
beta = 0.8;  # (kg/m^3) ppt^-1
k = 10e9  # (m^3/s)/(kg/m^3) #  3e-8; # sec^-1
# area of each of the ocean boxes, ridiculous value, needed to get
# all other results at the right order of magnitude:
area = 50000.0e3 ** 2
depth = 4000
V = area * depth  # volume of each of the ocean boxes
Sv = 1.e9  # m^3/sec

# # # -----------------------------------------------------------------------------

# # print("finding steady states, including unstable ones, as function of F_s or Lid")
# # # -----------------------------------------------------------------------------
Fs_range = np.arange(0, 10, 0.01) * area * S0 / year  # plausible range of freshwater flow rates to N. atlantic
# Lid_range=np.arange(0,1,0.01)
DeltaS_steady = np.zeros((3, len(Fs_range)))
q_steady = np.zeros((3, len(Fs_range)))
# DeltaS_steady=np.zeros((3,len(Lid_range)))
# q_steady=np.zeros((3,len(Lid_range)))

# Fs=2
Lid = 0.55  # 0.0 returns stommel

# test:
# Fs=Fs_func(0.0,0.0,False,Lid)
# #y=steady_states(Fs,alpha*DeltaT)
# y=steady_states_L(Fs,alpha*DeltaT,Lid)

i = 0
# Vary Fs for fixed Lid or vary Lid for fixed Fs
for Fs in Fs_range:
    y = steady_states_L(Fs, alpha * DeltaT, Lid)  # returns 3 solutions for beta*DeltaS
    DeltaS_steady[:, i] = y / beta
    for j in range(0, 2):  # 3 solutions
        q_steady[j, i] = q(DeltaT, DeltaS_steady[j, i], Lid)
    # print("Delta_S:", DeltaS_steady[0,i])  ##Check solution 1
    i = i + 1

# ##Varying Fs AND Lid

Fs_range_L = np.arange(0, 5, 0.01) * area * S0 / year
Lid_range_L = np.arange(0, 0.1, 0.05)
DeltaS_steady_L = np.zeros((3, len(Lid_range_L), len(Fs_range_L)))
q_steady_L = np.zeros((3, len(Lid_range_L), len(Fs_range_L)))
Fs_to_m_per_year = S0 * area / year  # Reference volume flux of salt

# Diagnostics
# print("Length of Lid_range_L:", len(Lid_range_L))
# print("Length of Fs_range_L:", len(Fs_range_L))
# print("Shape of DeltaS_steady_L:", DeltaS_steady_L.shape)
# print("Shape of q_steady_L:", q_steady_L.shape)

# nested loops: Find SS Delta_S and q for all combinations of Lid_L and Fs_L
for i, Lid_L in enumerate(Lid_range_L):
    for j, Fs_L in enumerate(Fs_range_L):
        y_L = steady_states_L(Fs_L, alpha * DeltaT, Lid_L)
        DeltaS_steady_L[:, i, j] = y_L / beta
        for k in range(3):  # 3 solutions
            q_steady_L[k, i, j] = q(DeltaT, DeltaS_steady_L[k, i, j], Lid_L)
    # print("Delta_S:", DeltaS_steady_L[0,i,j])  ##Check solution 1

##Still to do: IVP
# # print("calculating dDeltaS/dt as function of DeltaS for stability analysis...")
# # # ------------------------------------------------------------------------------
# # hysteresis=False
# # time=0
# # DeltaS_range=np.arange(-3,0,0.01)
# # rhs=np.zeros(len(DeltaS_range))
# # for i in range(0,len(DeltaS_range)):
# #     DeltaS = DeltaS_range[i]
# #     rhs[i]=rhs_S(time,DeltaS,time_max,DeltaT,hysteresis)
# #     i=i+1
# # rhs=rhs/np.std(rhs)

# # print("doing a hysteresis run...")
# # # --------------------------------
# hysteresis=True;
# y0=[0]
# teval=np.arange(0,time_max,time_max/1000)
# tspan=(teval[0],teval[-1])
# sol = solve_ivp(fun=lambda time,DeltaS: rhs_S(time,DeltaS,time_max,DeltaT,hysteresis) \
#                 ,vectorized=False,y0=y0,t_span=tspan,t_eval=teval)
# T=sol.t
# DeltaS=sol.y

# hysteresis=True;
# FWplot=np.zeros(len(T))
# qplot=np.zeros(len(T))
# i=0;
# for t in T:
#     FWplot[i]=Fs_func(t,time_max,hysteresis);
#     qplot[i]=q(DeltaT,DeltaS[0,i]);
#     i=i+1;

# N=len(qplot); N2=int(np.floor(N/2));


# ########################################################################
# ## plots:
# ########################################################################

print("testing form of q(L)")
Lidrange = np.arange(0, 1, 0.1)
DeltaS = -0.5
Lid_q = []

for ii in range(len(Lidrange)):
    Lid_q.append(Lid_gain_lookup(Lidrange[ii]))
# Lid_t = [q(DeltaT, DeltaS, value) for value in Lidrange]

Lid_q = np.array(Lid_q)

fig_num = 1
plt.figure(num=fig_num)
plt.plot(Lidrange, Lid_q, 'ro')
plt.xlabel('L = W/H');
plt.ylabel('Gain (-) on K')
plt.show()

fig_num = 2
plt.figure(num=fig_num)

print("plotting...")

# #SS with Fs as control -- classic stommel
Fs_to_m_per_year = S0 * area / year
plt.subplot(2, 3, 1)
plt.plot(Fs_range / Fs_to_m_per_year, DeltaS_steady[0, :], 'r.', markersize=1)
plt.plot(Fs_range / Fs_to_m_per_year, DeltaS_steady[1, :], 'g.', markersize=1)
plt.plot(Fs_range / Fs_to_m_per_year, DeltaS_steady[2, :], 'b.', markersize=1)
plt.plot(Fs_range / Fs_to_m_per_year, 0 * Fs_range, 'k--', dashes=(10, 5), linewidth=0.5)
plt.title('(a) Steady states', loc="left")
plt.xlabel('$F_s$ (m/year)');
plt.ylabel('$\Delta S$ (ppt)');
plt.xlim([min(Fs_range / Fs_to_m_per_year), max(Fs_range / Fs_to_m_per_year)])

plt.subplot(2, 3, 2)
plt.plot(Fs_range / Fs_to_m_per_year, q_steady[0, :] / Sv, 'r.', markersize=1)
plt.plot(Fs_range / Fs_to_m_per_year, q_steady[1, :] / Sv, 'g.', markersize=1)
plt.plot(Fs_range / Fs_to_m_per_year, q_steady[2, :] / Sv, 'b.', markersize=1)
plt.plot(Fs_range / Fs_to_m_per_year, 0 * Fs_range, 'k--', dashes=(10, 5), linewidth=0.5)
plt.title('(b) Steady states', loc="left")
plt.xlabel('$F_s$ (m/year)');
plt.ylabel('$q$ (Sv)');
plt.xlim([min(Fs_range / Fs_to_m_per_year), max(Fs_range / Fs_to_m_per_year)])
plt.show()
# # #SS with Lid governing Fs
# plt.subplot(2,3,1)
# plt.plot(Lid_range,DeltaS_steady[0,:],'r.',markersize=1)
# plt.plot(Lid_range,DeltaS_steady[1,:],'g.',markersize=1)
# plt.plot(Lid_range,DeltaS_steady[2,:],'b.',markersize=1)
# #plt.plot(Lid_range,0*Fs_range,'k--',dashes=(10, 5),linewidth=0.5)
# plt.title('(a) Steady states',loc="left")
# plt.xlabel('$L=W/H$ (-)');
# plt.ylabel('$\Delta S$ (ppt)');
# #plt.xlim([min(Fs_range/Fs_to_m_per_year), max(Fs_range/Fs_to_m_per_year)])

# plt.subplot(2,3,2)
# plt.plot(Lid_range,q_steady[0,:]/Sv,'r.',markersize=1)
# plt.plot(Lid_range,q_steady[1,:]/Sv,'g.',markersize=1)
# plt.plot(Lid_range,q_steady[2,:]/Sv,'b.',markersize=1)
# #plt.plot(Fs_range/Fs_to_m_per_year,0*Fs_range,'k--',dashes=(10, 5),linewidth=0.5)
# plt.title('(b) Steady states',loc="left")
# plt.xlabel('$L$=W/H (-)');
# plt.ylabel('$q$ (Sv)');
# plt.xlim([min(Fs_range/Fs_to_m_per_year), max(Fs_range/Fs_to_m_per_year)])

## SS with cariable Lid and Fs
# fig_num = 20
# plt.figure(num=fig_num)
fig, ax = plt.subplots(2, 3)

for i, Lid in enumerate(Lid_range_L):
    print(i)
    print("made it here!")
    # print(DeltaS_steady_L[0, i, :])
    ax[i, 0].plot(Fs_range_L / Fs_to_m_per_year, DeltaS_steady_L[0, i, :], 'r.', markersize=1)
    ax[i, 1].plot(Fs_range_L / Fs_to_m_per_year, DeltaS_steady_L[1, i, :], 'g.', markersize=1)
    ax[i, 2].plot(Fs_range_L / Fs_to_m_per_year, DeltaS_steady_L[2, i, :], 'b.', markersize=5)
    plt.show()
    # print(ax.ravel())
    # ax[i].plot(Fs_range_L / Fs_to_m_per_year, DeltaS_steady_L[i, 0, :], 'r.', markersize=1)
    # ax[0, i].plot(Fs_range_L / Fs_to_m_per_year, DeltaS_steady_L[i, 1, :], 'g.', markersize=1)
    # ax[0, i].plot(Fs_range_L / Fs_to_m_per_year, DeltaS_steady_L[i, 2, :], 'b.', markersize=1)
    # # plt.plot(Fs_range/Fs_to_m_per_year,0*Fs_range,'k--',dashes=(10, 5),linewidth=0.5)
    # ax[0].title('(a) Steady states', loc="left")
    # ax[0].xlabel('$F_s$ (m/year)');
    # ax[0].ylabel('$\Delta S$ (ppt)');
    # ax[0].xlim([min(Fs_range / Fs_to_m_per_year), max(Fs_range / Fs_to_m_per_year)])

    # # ax.subplot(2, 3, 2)
    # ax[1].plot(Fs_range_L / Fs_to_m_per_year, q_steady_L[0, i, :] / Sv, 'r.', markersize=1)
    # ax[1].plot(Fs_range_L / Fs_to_m_per_year, q_steady_L[1, i, :] / Sv, 'g.', markersize=1)
    # ax[1].plot(Fs_range_L / Fs_to_m_per_year, q_steady_L[2, i, :] / Sv, 'b.', markersize=1)
    # # plt.plot(Fs_range/Fs_to_m_per_year,0*Fs_range,'k--',dashes=(10, 5),linewidth=0.5)
    # ax[1].title('(b) Steady states', loc="left")
    # ax[1].xlabel('$F_s$ (m/year)');
    # ax[1].ylabel('$q$ (Sv)');
    # ax[1].xlim([min(Fs_range / Fs_to_m_per_year), max(Fs_range / Fs_to_m_per_year)])

# # plt.subplot(2,3,3)
# # plt.plot(DeltaS_range,rhs,'k-',lw=2)
# # plt.plot(DeltaS_range,rhs*0,'k--',dashes=(10, 5),lw=0.5)
# # # superimpose color markers of the 3 solutions
# # Fs=Fs_func(0.0,0.0,False)
# # yy=steady_states(Fs,alpha*DeltaT)/beta
# # plt.plot(yy[0],0,'ro',markersize=10)
# # plt.plot(yy[1],0,'go',markersize=10)
# # plt.plot(yy[2],0,'bo',markersize=10,fillstyle='none')
# # plt.title('(c) Stability',loc="left")
# # plt.xlabel('$\Delta S$ (ppt)');
# # plt.ylabel('$d\Delta S/dt$');

# plt.subplot(2,3,4)
# plt.plot(T[:N2]/year/1000,FWplot[:N2]/Fs_to_m_per_year,'b-',markersize=1)
# plt.plot(T[N2+1:]/year/1000,FWplot[N2+1:]/Fs_to_m_per_year,'r-',markersize=1)
# plt.plot(T/year/1000,0*T,'k--',dashes=(10, 5),linewidth=0.5)
# plt.xlabel('Time (kyr)');
# plt.ylabel('$F_s$ (m/yr)');
# plt.title('(d) $F_s$ for hysteresis run',loc="left");

# plt.subplot(2,3,5)
# plt.plot(T[:N2]/year/1000,qplot[:N2]/Sv,'b-',markersize=1)
# plt.plot(T[N2+1:]/year/1000,qplot[N2+1:]/Sv,'r-',markersize=1)
# plt.plot(T/year/1000,T*0,'k--',dashes=(10, 5),lw=0.5)
# plt.xlabel('Time (kyr)');
# plt.ylabel('MOC, $q$ (Sv)');
# plt.title('(e) MOC transport, $q$',loc="left");

# plt.subplot(2,3,6)
# plt.plot(FWplot[:N2]/Fs_to_m_per_year,qplot[:N2]/Sv,'b-',markersize=1)
# plt.plot(FWplot[N2+1:]/Fs_to_m_per_year,qplot[N2+1:]/Sv,'r-',markersize=1)
# plt.title('(f) $q$ hysteresis',loc="left");
# plt.xlabel('$F_s$ (m/yr)');
# plt.ylabel('$q$ (Sv)');


# # fig_num = 3
# # plt.figure(num=fig_num)
# # plt.subplot(2,3,2)
# # for ii in range(1, FWplot_L.shape[1]):
# #     plt.scatter(FWplot_L[:N2_L,ii]/Fs_to_m_per_year,qplot_L[:N2_L,ii]/Sv, s=1)
# #     plt.scatter(FWplot_L[N2_L+1:,ii]/Fs_to_m_per_year,qplot_L[N2_L+1:,ii]/Sv, s=1)

# # plt.title('(f) $q$ hysteresis',loc="left");
# # plt.xlabel('$F_s$ (m/yr)');
# # plt.ylabel('$q$ (Sv)');


# plt.tight_layout()
# plt.savefig("Figures/ocean-circulation-stommel-results.pdf")
# plt.show()

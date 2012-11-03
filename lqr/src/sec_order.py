# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate


k = 1
T = 1
ksi = 0.5
k1 = 1
k2 = 1




def system(x,t):
    dxdt = np.zeros_like(x)
    dxdt[0]=x[1]
    dxdt[1]=-x[0]*(k*k1+1/(T**2))-x[1]*(k*k2+2*ksi/T)
    
    
    return dxdt
    

def step(x,dt):
    x1_ = x[0] + x[1]*dt
    x2_ = (-x[0]*(k*k1 + 1/T**2)-x[1]*(k*k2+2*ksi/T))*dt+x[1]
    return np.array([x1_,x2_])

def update(intm,line_phase,line_time,x,t,dt):
    x = step(x,dt)
    t+=dt
    line_phase.set_xdata(x[0])
    line_phase.set_ydata(x[1])
    line_time.set_xdata(t)
    line_time.set_ydata(x[0])
    return line_phase

dt = 0.01
t = np.arange(0.0,20.0,dt)
x = np.array([1.,1.])

state = integrate.odeint(system, x, t)
state = np.transpose(state)
print len(t)
print len(state[0])



fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.axis([-5,5,-5,5])
line_phase, = ax1.plot(state[0],state[1])
ax1.set_ylabel(u"$x_2$")
ax1.set_xlabel(u"$x_1$") 
ax2 = fig.add_subplot(212)
ax2.axis([0,10,-5,5])
line_time, = ax2.plot(t,state[0])
ax2.set_xlabel(u"$t$")
ax2.set_ylabel(u"$x_1$")

#ani = animation.FuncAnimation(fig, update, fargs=(line_phase,line_time,x,t,dt), interval=25, blit=False)
plt.show()
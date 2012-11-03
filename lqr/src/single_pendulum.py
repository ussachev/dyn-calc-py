# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate

#Маятник с пропорциональным регулятором

m = 1.
l = 1.
g = 9.8
k1 = 1
k2 = 1
b = 100.

qd = np.pi*(-120.)/180.


def system(q,t):
    dqdt = np.zeros_like(q)
    u =(qd -q[0])*80.0
    dqdt[0]=q[1]
    dqdt[1]=(u-b*q[1]-m*g*l*np.sin(q[0]))/(m*l*2)
        
    return dqdt


dt = 0.01
t = np.arange(0.0,20.0,dt)
q0 = np.array([0.,0.])

q = integrate.odeint(system, q0, t)
q = np.transpose(q)
print len(t)
print len(q[0])

x = l*np.sin(q[0])
y = - l*np.cos(q[0])

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.grid()
ax1.set_aspect('equal', 'datalim')
ax1.axis([-5,5,-2,2])
pend_line, = ax1.plot([],[],'o-',lw=2)
ax2 = fig.add_subplot(212)
ax2.set_aspect('equal', 'datalim')
ax2.axis([-20,20,-10,10])
phase_line, = ax2.plot([],[],'-',lw=1)

time_template = 'time = %.1fs'
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

ax2.set_ylabel(u"$q_2$")
ax2.set_xlabel(u"$q_1$")
ax2.grid()

ax1.set_xlabel(u"$x$")
ax1.set_ylabel(u"$y$")

def init():
    pend_line.set_data([], [])
    phase_line.set_data([],[])
    time_text.set_text('')
    return pend_line,phase_line, time_text

def animate(i):
    thisx = [0., x[i]]
    thisy = [0., y[i]]

    pend_line.set_data(thisx, thisy)
    phase_line.set_data(q[0][:i],q[1][:i])
    time_text.set_text(time_template%(i*dt))
    return pend_line, phase_line, time_text

#ani = animation.FuncAnimation(fig, update, fargs=(line_phase,line_time,x,t,dt), interval=25, blit=False)
ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
    interval=25, blit=True, init_func=init)

plt.show()
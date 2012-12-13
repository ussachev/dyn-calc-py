# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate

#Двойной маятник с пропорциональным регулятором
m = np.array([0.5,0.1])
l = np.array([1.1,2.2])

g = 9.8
k1 = 1
k2 = 1
b1 = 10.
b2 = 10.

qd = np.pi*(-120.)/180.


def system(q,t0,m,l):
    dqdt = np.zeros_like(q)
    #u =(qd -q[0])*80.0
    u = np.array([0.0,0.0])
    ml1 = m[1]*l[1]**2
    m01l = (m[0]+m[1])*l[0]**2
    ml01 = 2*m[1]*l[0]*l[1]*np.cos(q[1])
    M_ = np.array([[ml1,-ml1-ml01],[-ml1-ml01,m01l+ml1+ml01]])/(ml1*(m01l+ml1+ml01)-(ml1+ml01)**2)
    V = np.array([-m[1]*l[0]*l[1]*(2*q[2]*q[3]+q[3]**2)*np.sin(q[1]),np.sin(q[1])*m[1]*l[0]*l[1]*q[2]**2])
    G = np.array([((m[0]+m[1])*g*l[0]*np.cos(q[0])+m[1]*g*l[1]*np.cos(q[0]+q[1])),m[1]*g*l[1]*np.cos(q[0]+q[1])])
    M_V = np.dot(M_,V)
    M_G = np.dot(M_,G)
    dqdt[0]=q[2]
    dqdt[1]=q[3]
    dqdt[2]=M_[0][0]*u[0]+M_[0][1]*u[1] - M_V[0] - M_G[0]
    dqdt[3]=M_[1][0]*u[0]+M_[1][1]*u[1] - M_V[1] - M_G[1]    
    return dqdt


dt = 0.01
t = np.arange(0.0,20.0,dt)
q0 = np.array([0.,0.,0.,0.])

q = integrate.odeint(system, q0, t,args=(m,l))
q = np.transpose(q)
print len(t)
print len(q[0])

x1 = l[0]*np.cos(q[0])
y1 = l[0]*np.sin(q[0])
x2 = x1+l[1]*np.cos(q[0]+q[1])
y2 = y1+l[1]*np.sin(q[0]+q[1])

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
    thisx = [0., x1[i],x2[i]]
    thisy = [0., y1[i],y2[i]]

    pend_line.set_data(thisx, thisy)
    phase_line.set_data(q[1][:i],q[3][:i])
    time_text.set_text(time_template%(i*dt))
    return pend_line, phase_line, time_text

#ani = animation.FuncAnimation(fig, update, fargs=(line_phase,line_time,x,t,dt), interval=25, blit=False)
ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y1)),
    interval=25, blit=True, init_func=init)

plt.show()
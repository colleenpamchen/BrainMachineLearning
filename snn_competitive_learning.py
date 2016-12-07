#!/bin/python
#-----------------------------------------------------------------------------
# File Name : brian2_lif.py
# Author: Emre Neftci
#
# Creation Date : Wed 28 Sep 2016 12:05:29 PM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
#Modified from Brian2 documentation examples 
from brian2 import *
from npamlib import *

Cm = 10*pF; gl = 1e-9*siemens; taus = 20*ms
Vt = 10*mV; Vr = 0*mV; #WTA section
#STDP Parameters
taupre = 20*ms; taupost = taupre
apre = 60e-12; apost = apre * taupre / taupost * 1.05  #Configuration and Fine Tuning Section

#--------------------------------------
#Discussion answers at bottom of script
#--------------------------------------
eqs = '''
dv/dt  = -gl*v/Cm + isyn/Cm : volt (unless refractory)
disyn/dt  = -isyn/taus : amp 
'''

data, labels = data_load_mnist([0,1]) #create 20 2d data samples
data = (data.T/(data.T).sum(axis=0)).T

duration = 200*ms
baseline_rate = 1000*Hz
## Spiking Network
#Following 2 lines for time-dependent inputs
rate = TimedArray(data*baseline_rate, dt = duration)
Pin = NeuronGroup(data.shape[1], 'rates = rate(t,i) : Hz', threshold='rand()<rates*dt')
P = NeuronGroup(9, eqs, threshold='v>Vt', reset='v = Vr',
                refractory=5*ms, method='euler')

Sff = Synapses(Pin, P, '''w : 1
                        dx/dt = -x / taupre  : 1 (event-driven)
                        dy/dt = -y / taupost : 1 (event-driven)''',
             on_pre='''isyn += w*amp
                        x += apre
                        w += y''',
             on_post='''y += apost
                        w += x - .01*w''')

Sff.connect() #connect all to all
Sff.w = '(rand()-.5)*20e-9'
#Learning Rule Section
wreci = -7 * nA
#Inhibitory connections
Sreci = Synapses(P, P, on_pre='isyn += wreci')
Sreci.connect()

#wrece = .02 * nA
#Excitatory connections
#Srece = Synapses(P, P, on_pre='isyn += wrece')
#Srece.connect(condition='abs(i-j)<=10') #connect to nearest 10

s_monin = SpikeMonitor(Pin)
s_mon = SpikeMonitor(P)

v_mon = StateMonitor(P, variables='v', record = [0])
isyn_mon = StateMonitor(P, variables='isyn', record = [0])

run(duration*len(data))

figure(figsize=(6,4))
plot(s_mon.t/ms, s_mon.i, '.k')
plot(s_monin.t/ms, s_monin.i, '.b', alpha=.2)
xlabel('Time (ms)')
ylabel('Neuron index')
ylim([-1,len(P)+1])
tight_layout()

figure()

W = np.array(Sff.w).T.reshape(784,9).T
stim_show(W-.5) #-.5 due to a plotting issue

show()

#-----------
#Discusssion
#-----------
#Several neurons learn the same digits partly because there are two images to learn
#and nine possible neurons to learn them.  If more than two neurons are trained, by definition at #least one will be a repeat.  The other part is that the images are disparate enough that
#it is easy to see the same numbers as different images, and are thus trained.

#Each learned digit is a specific weight map of the 784 inputs to the neuron.

#In mixture models, the population activity should be interpreted as the neurons
#responding to the features that they have learned through the training.  For example,
#the neurons that have learned features forthe numbers 8 and 9 will both respond 
#to the presence of a loop in the input.


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
# DISCUSSION: Several neurons learn the same digits because there are pattern 
# variations between data points within each digit class, and the competitive
# learning picks up on that variance. This is a generative model (not a classifier)
# of the data; it is clustering things that look similar. 
# These learned digits are internal representation of the spiking network, 
# after having learned the input. Each input digit has multiple classes of 
# neurons each representing a feature that the neuron has learned about the 
# digit during training. The mixture distribution describe the variations of 
# patterns within each class, where the weight matrix is the generative fields. 
# probability of k-class given stimulus. 

# [ ] investigate what the different state variables are doing


from brian2 import *
from npamlib import *

Cm = 9*pF; # lower this capacitance 
gl = 1e-9*siemens; 
taus = 29*ms # lengthens the amount of time the neurons fire 
Vt = 10*mV; Vr = 0*mV; 
#STDP Parameters: 
taupre = 50*ms; 
taupost = 50*ms;
apre = 1.0e-14; 
apost = -apre * taupre / taupost * 1.05 ;

tauca= .1 * second; 

eqs = '''
dv/dt  = -gl*v/Cm + isyn/Cm : volt (unless refractory)
dca/dt = -ca/tauca : 1
disyn/dt  = -isyn/taus : amp 
'''
# concentration of calcium ca approx. neuron's firing rate 

data, labels = data_load_mnist([0,1]) #create 20 2d data samples
data = (data.T/(data.T).sum(axis=0)).T # normalized inputs 
# [] concatenate data to itself 

duration = 200*ms
baseline_rate = 1000*Hz
## Spiking Network
#Following 2 lines for time-dependent inputs
rate = TimedArray(data*baseline_rate, dt = duration)
# input neurons 
Pin = NeuronGroup(data.shape[1], 'rates = rate(t,i) : Hz', threshold='rand()<rates*dt')
# n HIDDEN LAYER population neurons 
P = NeuronGroup(4 , eqs, threshold='v>Vt', reset='v = Vr;  ca+=1',
                refractory=5*ms, method='euler')               
# feedfoward synapses 
# R firing rate of post synaptic neuron, it is approximated by ca 
Rtarget = tauca * 35; 
P.ca = Rtarget /second ; 

a= 0.1; 


Sff = Synapses(Pin, P, '''dw/dt = a* w * (1-(ca/Rtarget) ) : 1
                        dx/dt = -x / taupre  : 1 (event-driven)
                        dy/dt = -y / taupost : 1 (event-driven)''',
             on_pre='''isyn += w*amp
                        x += apre
                        w = clip(w+y,0,1)  ''',
             on_post='''y += apost
                        w = clip(w+x,0,1) ''')         

Sff.connect() #connect all to all
Sff.w = '( rand() )*2e-12 ' # has to do with the feedforward synapse. change this to be higher. 

# recurrent 
# wreci = -5 * nA   ## VARIABLE OF IMPORTANCE. we want this to be high -7 
#Inhibitory connections

# recurrent inhibitory 
# Sreci = Synapses(P, P, on_pre='isyn += wreci')
# Sreci.connect() # all - to - all connect of inh. 
# i != j  

#wrece = .02 * nA
#Excitatory connections
#Srece = Synapses(P, P, on_pre='isyn += wrece')
#Srece.connect(condition='abs(i-j)<=10') #connect to nearest 10

s_monin = SpikeMonitor(Pin)
s_mon = SpikeMonitor(P)

v_mon = StateMonitor(P, variables=['v','ca'], record = [0])
isyn_mon = StateMonitor(P, variables='isyn', record = [0])
w_mon = StateMonitor(Sff, variables='w', record=[0])


#run(duration*len(data)/40) # run for 200 samples 
run(duration*200) # run for 200 samples 

figure(figsize=(6,4))
plot(s_mon.t/ms, s_mon.i, '.k')
plot(s_monin.t/ms, s_monin.i, '.b', alpha=.2)
xlabel('Time (ms)')
ylabel('Neuron index')
ylim([-1,len(P)+1])
tight_layout()

#figure(figsize=(6,4))
#ax = axes()
#ax2 = ax.twinx()
#ax.plot(v_mon.t/ms, v_mon.v[0]/mV, 'k')
#ax.set_xlabel('Time (ms)')
#ax.set_ylabel('Membrane potential [mV]')
#ax2.plot(isyn_mon.t/ms, isyn_mon.isyn[0]/nA, 'b', linewidth = 3, alpha=.4)
#ax2.set_xlabel('Time (ms)')
#ax2.set_ylabel('Synaptic Current [nA]')
#tight_layout()
figure()
W = np.array(Sff.w).T.reshape(784,len(P)).T
stim_show(W-.5) #-.5 due to a plotting issue

# Plot calcium: 
figure(figsize=(6,4))
plot(v_mon.t/ms, v_mon.ca.T,'.r')
xlabel('Time (ms)')

# Plot weights: 
figure(figsize=(6,4))
plot(v_mon.t/ms, w_mon.w.T,'.r')
xlabel('Time (ms)')

show()













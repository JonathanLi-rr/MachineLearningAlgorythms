import numpy as np
import MDP
import RL
import matplotlib.pyplot as plt

''' Construct simple MDP as described in Lecture 16 Slides 21-22'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9        
mdp = MDP.MDP(T,R,discount)
rlProblem = RL.RL(mdp,np.random.normal)

# Test Q-learning 
cdrc=np.zeros(200)
for trial in range (3):
  [Q,policy], CDR = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.3)
  CDRnp = np.array(CDR)
  cdrc += CDRnp
cdrc/=3
plt.plot(np.arange(200), cdrc)
plt.show()



#print("\nQ-learning results")
#print(Q)
#print(policy)

import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''


        Q = np.copy(initialQ)
        policy = np.argmax(Q, axis=0)
        nact = self.mdp.nActions
        nstate = self.mdp.nStates
        CDR = []

        counter = np.zeros([nstate, nact])
        for e in range(nEpisodes):
          s = s0
          cdre=0
          for step in range(nSteps):
            if np.random.rand() < epsilon:
              a= np.random.randint(0, nact)
            else:
              if temperature == 0:
                a = np.argmax(Q[:, s])
              else:
                  p = np.exp(Q[:, s] / temperature)
                  p /= np.sum(p)
                  a = np.random.choice(nact, p=p)

            reward, nextState= self.sampleRewardAndNextState(s, a)
            cdre+= reward*(self.mdp.discount**step)
            
            counter[s, a]+=1
            learnR = 1/counter[s, a]
            
            scoreChange = learnR*(reward+self.mdp.discount*np.max(Q[:, nextState])-Q[a, s])
            Q[a, s]+=scoreChange
            s = nextState
            policy = np.argmax(Q, axis=0)
          CDR.append(cdre)
        

        

        return [Q,policy], CDR    
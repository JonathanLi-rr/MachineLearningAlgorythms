import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        
        V = initialV.copy()
        newV = initialV.copy()
        iterId = 0
        epsilon = np.inf

        while (iterId<nIterations and epsilon>tolerance):
          for s in range (self.nStates):
            tarr = [0]*self.nActions
            for a in range(self.nActions):
              Ra = self.R[a, s]
              Ta = self.T[a, s, :]
              tarr[a]  = Ra + self.discount*np.dot(Ta, V)

            maxpick = max(tarr)
            newV[s] = maxpick
          epsilon = np.linalg.norm(V-newV)  
          iterId+=1
          V = newV.copy()
              
        
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        policy = np.zeros(self.nStates)
        for s in range (self.nStates):
          tarr = [0]*self.nActions
          for a in range(self.nActions):
            Ra = self.R[a, s]
            Ta = self.T[a, s, :]
            tarr[a]  = Ra + self.discount*np.dot(Ta, V)
          maxindex = tarr.index(max(tarr))
          policy[s]=int(maxindex)

        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        A = np.zeros((self.nStates, self.nStates))
        b = np.zeros(self.nStates)
        for s in range (self.nStates):
          a = int(policy[s])

          b[s] = self.R[a, s]

          
          Ta = self.T[a, s, :]
          A[s] = Ta

        A = self.discount * A - np.identity(self.nStates)
        V= np.linalg.solve(A, -b)
          #V[s] = self.R[a, s] + self.discount * np.dot(Ta, V)

        

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = initialPolicy.copy()
        V = np.zeros(self.nStates)
        iterId = 0


        while iterId < nIterations:
          V = self.evaluatePolicy(policy)
          new_policy = self.extractPolicy(V)
          iterId+=1
          if np.array_equal(new_policy, policy):
            break
          policy = new_policy.copy()
          
            
        return [policy,V,iterId]



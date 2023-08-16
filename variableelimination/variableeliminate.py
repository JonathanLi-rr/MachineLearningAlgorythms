import numpy as np

'''
Implement the variable elimination algorithm by coding the
following functions in Python. Factors are essentially 
multi-dimensional arrays. Hence use numpy multidimensional 
arrays as your main data structure.  If you are not familiar 
with numpy, go through the following tutorial: 
https://numpy.org/doc/stable/user/quickstart.html
'''



######### restrict function
# Tip: Use slicing operations to implement this function
#
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be restricted
# value -- integer indicating the value to be assigned to variable
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been restricted to value)
#########
def restrict(factor,variable,value):

	slices=tuple(slice(None) if i!=variable else slice(value,value+1) for i in range(len(factor.shape)))
	resulting_factor=factor[slices]
	return resulting_factor

######### sumout function
# Tip: Use numpy.sum to implement this function
#
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be summed out
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been summed out)
#########
def sumout(factor,variable):

	resulting_factor = np.sum(factor, axis=variable, keepdims=True)
	return resulting_factor

######### multiply function
# Tip: take advantage of numpy broadcasting rules to multiply factors with different variables
# See https://numpy.org/doc/stable/user/basics.broadcasting.html
#
# Inputs: 
# factor1 -- multidimensional array (one dimension per variable in the domain)
# factor2 -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (elementwise product of the two factors)
#########
def multiply(factor1,factor2):

	resulting_factor = factor1*factor2
	return resulting_factor

######### normalize function
# Tip: divide by the sum of all entries to normalize the factor
#
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (entries are normalized to sum up to 1)
#########
def normalize(factor):

	
	resulting_factor = factor/np.sum(factor)
	return resulting_factor

######### inference function
# Tip: function that computes Pr(query_variables|evidence_list) by variable elimination.  
# This function should restrict the factors in factor_list according to the
# evidence in evidence_list.  Next, it should sumout the hidden variables from the 
# product of the factors in factor_list.  The variables should be summed out in the 
# order given in ordered_list_of_hidden_variables.  Finally, the answer should be
# normalized to obtain a probability distribution that sums up to 1.
#
#Inputs: 
#factor_list -- list of factors (multidimensional arrays) that define the joint distribution of the domain
#query_variables -- list of variables (integers) for which we need to compute the conditional distribution
#ordered_list_of_hidden_variables -- list of variables (integers) that need to be eliminated according to thir order in the list
#evidence_list -- list of assignments where each assignment consists of a variable and a value assigned to it (e.g., [[var1,val1],[var2,val2]])
#
#Output:
#answer -- multidimensional array (conditional distribution P(query_variables|evidence_list))
#########
def inference(factor_list,query_variables,ordered_list_of_hidden_variables,evidence_list):

  for v in query_variables:
    if v in ordered_list_of_hidden_variables:
      ordered_list_of_hidden_variables.remove(v)

  for pair in evidence_list:
    var=pair[0]
    val=pair[1]
    for i in range (len(factor_list)):
      if factor_list[i].shape[var]>1:
        newf = restrict(factor_list[i], var, val)
        factor_list[i]=newf
  
  for var in ordered_list_of_hidden_variables:
    elim = []
    for i in range (len(factor_list)):
      if factor_list[i].shape[var]>1:
        elim.append(i)
    if len(elim)>0:
      newf = factor_list[elim[0]]
      for i in range(1, len(elim)):
        newf = multiply(newf, factor_list[elim[i]])

      elim.sort(reverse=True)
      for ind in elim:
        del factor_list[ind]
      newf=sumout(newf, var)
      factor_list.append(newf)

  answer = factor_list[0]
  for i in range (1, len(factor_list)):
    answer = multiply(answer, factor_list[i])
  answer=normalize(answer)
  return answer


####################
###Calculating queestions from assignment







#########
##q2b
T=0
F=1
A=2
P=3
O=4
V=5

f1 = np.array([[0.996,0.004],[0.99, 0.01]])
f1 = f1.reshape(2, 2, 1, 1, 1, 1)

f2=np.array([0.95, 0.05])
f2=f2.reshape(2, 1, 1, 1, 1, 1)

f3=np.array([[[0.99, 0.01], [0.9, 0.1]],[[0.1, 0.9], [0.1, 0.9]]])
f3=f3.reshape(2,2, 1,2, 1, 1)

f4=np.array([0.2, 0.8])
f4=f4.reshape(1,1, 2, 1,1,1)

f5=np.array([[[0.9, 0.1], [0.7, 0.3]],[[0.4, 0.6], [0.2, 0.8]]])
f5=f5.reshape(1,2, 2, 1, 2, 1)

f6 = np.array([[0.99,0.01],[0.9, 0.1]])
f6=f6.reshape(1, 1, 2, 1,1, 2)


f7 = inference([f1,f2,f3,f4, f5, f6],[F],[T, P, O, A, V],[])
print("q2b first query", f7)

f8 = inference([f1,f2,f3,f4, f5, f6],[F],[T, P, O, A, V],[[P, 1], [O, 0], [V,1]])
print("q2b second query", f8)

f9 = inference([f1,f2,f3,f4, f5, f6],[F],[T, P, O, A, V],[[P, 1], [O, 0], [V,1], [T, 1]])
print("q2c query", f9)

f10 = inference([f1,f2,f3,f4, f5, f6],[F],[T, P, O, A, V],[[P, 0], [T, 0],[A, 1], [O,1]])
print("q2d query", f10)
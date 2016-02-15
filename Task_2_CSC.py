# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 21:27:15 2016

@author: Charles
"""

from __future__ import division
import numpy as np
from numpy.linalg import lstsq

#-----------------the sampled data used is taken from file name 'Task_1_CSC'--------

#create a matrix of sampled output values       
Y = [[0.0],
      [0.18292719311245312],
      [0.33239202824490549],
      [0.45451568061756265],
      [0.55429959604904888], 
      [0.63583031991288275],
      [0.70244685730787915], 
      [0.75687741850233803],
      [0.80135114991795886],
      [0.8376894264784831], 
      [0.86738044410524662], 
      [0.89164016721689399], 
      [0.91146212727404219], 
      [0.92765811181594937], 
      [0.94089141036591251], 
      [0.95170397875651236], 
      [0.96053863436108289], 
      [0.96775719121379411],
      [0.97365527772311644], 
      [0.97847444382255377], 
      [0.98241205339427828], 
      [0.98562936709947491], 
      [0.98825814663921729],
      [0.99040605091644318], 
      [0.99216104509316194], 
      [0.99359500311120508],
      [0.99476665121396624], 
      [0.99572397301797411], 
      [0.99650617463146929], 
      [0.99714529029935928]]

      

#create a list of inputs according to the difference equation          
X = [[0,0],
     [ 0.          ,1        ],
     [ 0.18292719  ,1.        ],
     [ 0.33239203  ,1.        ],
     [ 0.45451568  ,1.        ],
     [ 0.5542996   ,1.        ],
     [ 0.63583032  ,1.        ],
     [ 0.70244686  ,1.        ],
     [ 0.75687742  ,1.        ],
     [ 0.80135115  ,1.        ],
     [ 0.83768943  ,1.        ],
     [ 0.86738044  ,1.        ],
     [ 0.89164017  ,1.        ],
     [ 0.91146213  ,1.        ],
     [ 0.92765811  ,1.        ],
     [ 0.94089141  ,1.        ],
     [ 0.95170398  ,1.        ],
     [ 0.96053863  ,1.        ],
     [ 0.96775719  ,1.        ],
     [ 0.97365528  ,1.        ],
     [ 0.97847444  ,1.        ],
     [ 0.98241205  ,1.        ],
     [ 0.98562937  ,1.        ],
     [ 0.98825815  ,1.        ],
     [ 0.99040605  ,1.        ],
     [ 0.99216105  ,1.        ],
     [ 0.993595    ,1.        ],
     [ 0.99476665  ,1.        ],
     [ 0.99572397  ,1.        ],
     [ 0.99650617  ,1.        ]]


     
# transpose of matrix X
Xt =  np.matrix.transpose(np.array(X))        

#multiplication of matrix     
mat1 = np.dot(Xt, X)
mat2 = np.dot(Xt,Y)

#identity matrix
I = np.eye(len(mat1))

#solve for the inverse matrix of mat1 above
pre_beta = lstsq(mat1, I)[0]

#solving the list of process parameters according to the difference equation
beta = np.dot(pre_beta, mat2)
print beta
#solve for the time constant
tau = -1/np.log(beta[0][0])
print 'tau =',tau
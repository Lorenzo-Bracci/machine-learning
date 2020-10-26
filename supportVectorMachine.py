import numpy , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

numpy.random.seed(100) # only here for debugging purposes to make the datasets always the same

classA = numpy.concatenate (
(numpy . random.randn(10 , 2) * 0.2 + [ 1.5 , 0.5 ] ,
numpy.random.randn (10 , 2) * 0.2 + [ -1.5 , 0.5 ] ) )
classB = numpy.random.randn (20 , 2) * 0.2 + [ 0.0 , -0.5]
inputs = numpy.concatenate ( ( classA , classB ) )
targets = numpy.concatenate (
(numpy.ones( classA.shape[0] ) ,
-numpy.ones( classB.shape[0] ) ) )
N = inputs.shape[0] # Number of rows ( samples )
permute=list( range(N) )
random.shuffle(permute)
inputs = inputs[permute , : ]
targets = targets[permute]



def linearKernel(vector1, vector2):
    return numpy.dot(vector1, vector2)

def polynomialKernel(vector1, vector2, i):
    res = (numpy.dot(vector1, vector2) + 1)**i
    return res

def radialKernel(vector1, vector2, sigma):
    return math.exp(-(numpy.linalg.norm(vector1 - vector2)**2)/(2*sigma**2))

def kernel(vector1, vector2):
    return linearKernel(vector1, vector2)

def objective(paramsVector):
    sum = 0
    for i in range(N):
       for j in range(N):
          currentIteration = paramsVector[i]*paramsVector[j]*targets[i]*targets[j]*kernel(inputs[i], inputs[j])
          sum = sum + currentIteration
    sum = sum / 2
    for x in range(N):
        sum = sum - paramsVector[x]
    return sum


def zeroFun(paramsVector):
   sum = 0
   for i in range(N):
      sum = sum + (paramsVector[i]*targets[i])
   return sum

ret = minimize( objective, numpy.zeros(N),
bounds=[(0,1) for b in range(N)], constraints={'type':'eq', 'fun':zeroFun} )
alpha = ret['x']

def computeB(supportVectorIndex):
    res = 0
    for i in range(N):
        res = res + (alpha[i]*targets[i]*kernel(inputs[supportVectorIndex], inputs[i]) )
    return res - targets[supportVectorIndex]


b = 0

for i in range(N):
    if(alpha[i] > 10e-5 ):
        b = computeB(i)

def indicator(currentPointX, currentPointY):
    res = 0
    for i in range(N):
        res = res + (alpha[i]*targets[i]*kernel([currentPointX, currentPointY], inputs[i]) )
    return res - b


plt.plot([p[0] for p in classA ] ,
[ p[1] for p in classA ] ,
'b.' )
plt.plot([ p[0] for p in classB ] ,
[ p[1] for p in classB ] ,
'r.' )
plt.axis('equal') # Force same scale on both axes
xgrid=numpy.linspace(-5, 5)
ygrid=numpy.linspace(-4, 4)
grid=numpy.array([ [ indicator(x , y )
for x in xgrid ]
for y in ygrid ] )
plt.contour( xgrid , ygrid , grid ,
( -1.0 , 0.0 , 1.0 ) ,
colors =( 'red' , 'black' , 'blue' ) ,
linewidths =(1 , 3 , 1))

plt.savefig('svmplot.pdf') # Save a copy in a file
plt.show() # Show the plot on the screen

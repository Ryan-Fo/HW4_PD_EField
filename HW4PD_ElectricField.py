import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def distance(xf,xi,yf,yi):
    """
    Standard distance formula between two points using pythagorian theorem
    Inputs: final x value, inital x value, final y, inital y
    Outputs: distance / radius to given point
    """
    distance = np.sqrt((xf-xi)**2 + (yf-yi)**2)
    return distance

def epotential(r):
    """
    Calculates electrical potential given a distance value
    Inputs: radius between two charges
    Outputs: electrical potential at radius distance 
    """
    vpepsilon = (8.85418782 * (10**-12))
    potential = (1 / (4 * np.pi * vpepsilon * r))
    return potential

def partial_deriv_x (x,y,h=0.01,func = None):
    """
    Calculates partial derivative with respect to x [df/dx]
    Inputs: x and y
    Output: Partial derivaive of x keeping y constant
    """
    return (g(x+h,y)-g(x-h,y))/(2*h)

def partial_deriv_y (x,y,h=0.01, func = None):
    """
    Calculates partial derivative with respect to y [df/dy]
    Inputs: x and y
    Output: Partial derivaive of y keeping x constant
    """
    return (g(x,y+h)-g(x,y-h))/(2*h)
#interpolation
def f(x):
    """Function of x to be tested with"""
    return np.sqrt(x) + np.cos(x)
    
def analytical_derivative_f(x):
    """True derivative to test with"""
    return 1/(2*sqrt(x)) - np.sin(x)
    
def central_difference(x, h, func=None):
    """Computes the numerical derivative
    of an arbitrary function using the central
    difference method.
    x: point at which to evaluate the func
    h: stepsize to compute the secant
    func: a valid python function"""
    numerator = func(x+h)-func(x-h)
    denominator = (2 * h)
    return numerator/denominator

if __name__ == "__main__":
    
    #Charge 1 at 0.05 x and 0 y
    #Charge 2 at -0.05 x and 0 y    
    
    spacing = 0.01
    x_range = np.arange(-0.5, 0.51, spacing)
    y_range = np.arange(-0.5, 0.51, spacing)
    xs, ys  = np.meshgrid(x_range, y_range)
    
    print('X coordinates')
    print(xs)
    print('Y coordinates')
    print(ys)
    print('\n')
    
    """
    #attempt to find partial derivatives for charge potential of x and y
    ydistancelist1 = []
    ydistancelist2 = []
    xdistancelist1 = []
    xdistancelist2 = []
    
    for x in xs:
        for y in ys:
            point = (x,y)
            distance_x1 = -0.05 - x
            distance_x2 = 0.05 - x
            distance_y = 0 - y
    
    epo1x = epotential(distance_x1)
    epo2x = epotential(distance_x2)
    epo1y = epotential(distance_y)
    epo2y = epotential(distance_y)
    
    sumex = epo1x + epo2x
    sumey = epo1y + epo2y
    
    h = 0.01
    partial_deriv_x(sumex,sumey,h,f)
    partial_deriv_y(sumex,sumey,h,f)
    
    
    dis_from_c1 = distance(-0.05,xs,0.0,ys)
    print(dis_from_c1)

    dis_from_c2 = distance(0.05,xs,0.0,ys)
    print(dis_from_c2)
    """
    
    charge_potential1 = epotential(dis_from_c1)
    charge_potential2 = epotential(dis_from_c2)
    potentialsum = charge_potential1 + charge_potential2
    
    potentialsum[potentialsum>3.5*(10**11)]=None #reduces huge values to see shape of range
    print('Potential Sum:',potentialsum)
    print('\n')

    contour = plt.contour(potentialsum)
    plt.title('Electric Potential around 2 charges')
    plt.xlabel('Distance in x from charges')
    plt.ylabel('Distance in y from charges')
    plt.colorbar()
    plt.show()
    
    
    
    #part 2: Interpolation exploration
    sparse_x = np.arange(0,4*np.pi,0.8)
    sparse_values=[]
    for x_value in np.arange(0,4*np.pi,0.8):
        deriv = f(x_value)
        sparse_y = central_difference(x_value,0.8,f)#attempted to calculate similar to previous hw.
        sparse_values.append(sparse_y)

    sparse_y = f(sparse_values)
    print(sparse_y)
    sparse_y[sparse_y>3] = None
    f = interp1d(sparse_x,sparse_values, kind = 'linear')
    f2 = interp1d(sparse_x,sparse_values, kind ='cubic')
    sparsenew = np.arange(0,12)
    
    plt.plot(sparse_x,sparse_y,'o',sparsenew,f(sparsenew),'-',sparsenew,f2(sparsenew),'--')
    plt.legend(['data','linear','cubic'],loc = 'best')
    plt.show()
    fine_x = np.arange(0,4*np.pi,0.1)
    
    

    

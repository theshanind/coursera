import numpy as np

def gradiant_descent(x,y):  # Note the spelling "gradiant" with an 'a'
    m_curr = b_curr = 0
    iteration = 10000
    n = len(x)
    learning_rate = 0.08
    for i in range(iteration):
        y_prediced = m_curr * x + b_curr
        cost = (1/n)* sum ([val **2 for val in (y - y_prediced)])
        md = -(2/n)*sum(x*(y-y_prediced))
        bd = -(2/n)*sum(y-y_prediced)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd  # Fixed: should use bd here, not md
        print('m {},b {},cost{} iteration{}'.format(m_curr,b_curr,cost,i,))

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradiant_descent(x,y)
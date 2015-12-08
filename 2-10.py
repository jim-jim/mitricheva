from random import gauss
import numpy as np
import cvxpy as cvx
m = 12 #number of balls
n = 4  #dimension of space
def rand_ball(): return (np.random.uniform(0,4,n), 4) #center and radius
balls = [rand_ball() for i in range(m)] #you can specify your own balls here
box = max(ball[1]+abs(ball[0][i]) for i in range(n) for ball in balls) #half-size of the first container
center = cvx.Variable(n)
obj_func = sum(cvx.log(box + j * center[i]) for j in (-1,1) for i in range(n)) #logarithmic barriers
constr = [j * center[i] <= box for j in (-1,1) for i in range(n)] #define the domain of the objective
def check_center(val):
    global obj_func, constr
    for ball in balls:
        diff = np.subtract(val, ball[0])
        if np.linalg.norm(diff) > ball[1]: #current point doesn't belong to this ball
            constr.append(diff * center <= 0) #add cutting plane
            obj_func += cvx.log(-diff * center)
            return False
    return True #common point was found
ok = -1
val = np.zeros(n) #first iteration is obvious
for i in range(100): #prevent infinite loop
    if i > 0: val = np.asarray(np.transpose(center.value))[0]
    if check_center(val): ok = 1; break #yeah!
    obj = cvx.Maximize(obj_func)
    prob = cvx.Problem(obj, constr)
    prob.solve()
    if prob.status != cvx.OPTIMAL and prob.status != cvx.OPTIMAL_INACCURATE: ok = 0; break #no matter, accurate or not
if ok == 1: print("Common point: " + str(val))
elif ok == 0: print("Intersection is empty")
else: print("Not enough iterations")
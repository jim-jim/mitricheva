import numpy as np
import cvxpy as cvx
m = 7 #number of balls
n = 4  #dimension of space, >= 2
def rand_ball(): return (np.random.uniform(0,4,n), 3) #center and radius
balls = [rand_ball() for i in range(m)] #you can specify your own balls here
box = max(ball[1]+abs(ball[0][i]) for i in range(n) for ball in balls) #half-size of the first container
center = cvx.Variable(n)
obj_func = sum(cvx.log(box + j * center[i]) for j in (-1,1) for i in range(n)) #logarithmic barriers
constr = [j * center[i] <= box for j in (-1,1) for i in range(n)] #define the domain of the objective
def check_center(val):
    global obj_func, constr
    res = True #common point was found
    for ball in balls:
        diff = np.subtract(ball[0], val)
        if res:
            diff_len = np.linalg.norm(diff)
            if diff_len > ball[1]: #current point doesn't belong to this ball
                p = sum(diff * val)
                constr.append(diff * center >= p) #add cutting plane
                obj_func += cvx.log(diff * center - p)
                res = False #next we check if this plane intersects with other balls
        else:
            if sum(diff * val) - p + diff_len * ball[1] < 0: return None #no intersection
    return res
val = np.zeros(n) #first iteration is obvious
for i in range(100): #prevent infinite loop
    ok = check_center(val)
    if ok != False: break #yeah!
    obj = cvx.Maximize(obj_func)
    prob = cvx.Problem(obj, constr)
    prob.solve()
    if prob.status != cvx.OPTIMAL and prob.status != cvx.OPTIMAL_INACCURATE: ok = None; break #no matter, accurate or not
    val = np.asarray(np.transpose(center.value))[0] #convert column to row
if ok: print("Common point: " + str(val))
elif ok == None: print("Intersection is empty")
else: print("Not enough iterations")
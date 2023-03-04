import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import pickle
import optimizer

with open("first_database.pkl", "rb") as f:
    x,y = pickle.load(f)
n= 100
m = 10

#mélange et séléction des données
X = x[:n]
Y = y[:n]
stack = np.stack([X,Y]).T
np.random.shuffle(stack)
x_shuffle = stack.T[0]
y_shuffle = stack.T[1]
x_selected = np.array([x_shuffle[10*i] for i in range(10)])
agents_x = np.array([[0,1],[2,3],[4,5],[6,7],[8,9]])
agents_y = np.array([np.arange(20*i,20*i+20) for i in range(5)])

#matrices stochastiques (pour DGD et gradient_tracking)
W = (1/3)*np.array([[1,1,0,0,1],[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[1,0,0,1,1]])
W1 = (1/3)*np.array([[2,1,0,0,0],[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,0,1,2]])
W1_1 = np.ones((5,5))/5

#matrices d'adjacence pour dual decomposition et la suite
W2 = np.ones((5,5)) 
W3 = np.array([[1,1,0,0,1],[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[1,0,0,1,1]]) #cycle
W4 = np.array([[1,1,0,0,1],[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,0,1,1]])
W5= np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,1]])

#création de l'instance du problème
instance_pb = optimizer.instance(x_shuffle,x_selected,y_shuffle,agents_x,agents_y,W2,0.5)

#Solvers

solver_non_cooperative = optimizer.gradient_descent(instance_pb,0.0001,20000,np.ones(10))
solver_non_cooperative.solve()
print(solver_non_cooperative.curent_solution)

# solver_DGD = optimizer.DGD(instance_pb,0.001,200,np.ones(10))
# solver_DGD.solve()
# print(solver_DGD.curent_solution)

# solver_gradient_tracking = optimizer.gradient_tracking(instance_pb,0.0001,1000,np.ones(10))
# solver_gradient_tracking.solve()
# print(solver_gradient_tracking.curent_solution)

# solver_dual_decomposition = optimizer.dual_decomposition(instance_pb,0.001,0.001,10,100,np.ones(10),np.ones(50))
# solver_dual_decomposition.solve()
# print(solver_dual_decomposition.curent_primal_solution)

# solver_dual_decomposition_edge = optimizer.dual_decomposition_edge(instance_pb,0.0001,0.001,100,100,np.ones(10),np.ones(10))
# solver_dual_decomposition_edge.solve()
# print(solver_dual_decomposition_edge.curent_primal_solution)

solver_ADMM = optimizer.ADMM(instance_pb,0.001,300,np.ones(10),np.ones(10),500)
solver_ADMM.solve()
print(solver_ADMM.curent_solution)



import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import pickle
import optimizer

# def main():
with open("first_database.pkl", "rb") as f:
    x,y = pickle.load(f)
n= 100
m = 10
sel = [i for i in range(n)]
ind = np.random.choice(sel, m, replace=False)
x_selected = [x[i] for i in ind]
print(x_selected)
y_selected = [y[i] for i in ind]
agents = np.array([[0,1],[2,3],[4,5],[6,7],[8,9]])
W = (1/3)*np.array([[1,1,0,0,1],[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[1,0,0,1,1]])
W1 = (1/3)*np.array([[2,1,0,0,0],[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,0,1,2]])

W2 = np.ones((5,5)) # pour la dual decomposition, W est une matrice d'adjacence (pas stochastique)
instance_pb = optimizer.instance(x_selected,y_selected,agents,W2,0.5)

# solver_DGD = optimizer.DGD(instance_pb,0.001,1000,7*np.ones(10))
# solver_DGD.solve()
# print(solver_DGD.curent_solution)

# solver_gradient_tracking = optimizer.gradient_tracking(instance_pb,0.001,1000,7*np.ones(10))
# solver_gradient_tracking.solve()
# print(solver_gradient_tracking.curent_solution)

solver_dual_decomposition = optimizer.dual_decomposition(instance_pb,0.01,0.01,1000,200,np.ones(10),np.ones(50))
solver_dual_decomposition.solve()
print(solver_dual_decomposition.curent_primal_solution)

# if __name__ == "__main__":
#     main()

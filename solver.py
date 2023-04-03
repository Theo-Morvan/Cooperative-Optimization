import numpy as np
import pandas as pd


def kernel(x, y):
    return np.exp(-((x - y) ** 2))


def compute_doubly_stochastic_matrix(adjacence_matrix):
    n, p = np.shape(adjacence_matrix)
    doubly_stochastic_matrix = np.copy(adjacence_matrix)
    max_sum = np.max(np.array([np.sum(adjacence_matrix[i]) for i in range(n)]))
    for i in range(n):
        doubly_stochastic_matrix[i, i] = max_sum - (np.sum(adjacence_matrix[i]) - 1)
    return doubly_stochastic_matrix / max_sum



class instance:
    def __init__(self,X,centers_kernel,Y,kernel_center_of_agent,data_point_of_agent,communication_matrix,sigma) -> None:
        self.centers_kernel=centers_kernel
        self.X = X
        self.Y=Y
        self.kernel_center_of_agent = kernel_center_of_agent
        self.data_point_of_agent = data_point_of_agent
        self.communication_matrix = communication_matrix
        self.sigma = sigma
        number_of_kernels = np.shape(centers_kernel)[0]
        number_of_datas = np.shape(X)[0]
        self.number_of_datas = number_of_datas
        self.number_of_kernels = number_of_kernels
        self.Kmm_matrix = np.array([[kernel(centers_kernel[k],centers_kernel[l]) for k in range(number_of_kernels)]for l in range(number_of_kernels)])
        self.Knm_matrix = np.array([[kernel(centers_kernel[k],X[l]) for k in range(number_of_kernels)]for l in range(self.number_of_datas)])
        number_of_agents,kernels_per_agent = np.shape(kernel_center_of_agent)
        _,data_per_agent = np.shape(data_point_of_agent)
        self.number_of_agents = number_of_agents
        self.kernels_per_agent = kernels_per_agent
        self.data_per_agent = data_per_agent
        matrixA = np.identity(number_of_agents*number_of_kernels)
        for index in range(number_of_kernels):
                matrixA[(number_of_agents-1)*number_of_kernels+index,(number_of_agents-1)*number_of_kernels+index]=0
        for index in range((number_of_agents-1)*number_of_kernels):
                matrixA[index,index+number_of_kernels]=-1
        self.matrixA = matrixA
        self.true_solution = np.zeros(number_of_agents)

    def objective(self,curent_solution):
        objecti = np.dot(curent_solution,np.dot(self.Kmm_matrix,curent_solution))/2
        for agent_index in range(self.number_of_agents):
            for agent_index_2 in range(self.data_per_agent):
                matrix_Kim = np.array([kernel(self.X[self.data_point_of_agent[agent_index,agent_index_2]],self.centers_kernel[j]) for j in range(self.number_of_kernels)])
                objecti+=(1/(2*(self.sigma**2)))*(self.Y[self.data_point_of_agent[agent_index,agent_index_2]]-np.dot(matrix_Kim,curent_solution))**2
        return(objecti)
    
    def gradient(self,agent_index,point):
        sum  = np.dot(self.Kmm_matrix,point)/5    
        for agent_index_2 in range(self.data_per_agent):
            matrix_Kim = np.array([kernel(self.X[self.data_point_of_agent[agent_index,agent_index_2]],self.centers_kernel[j]) for j in range(self.number_of_kernels)])
            sum -=matrix_Kim*(self.Y[self.data_point_of_agent[agent_index,agent_index_2]]-np.dot(matrix_Kim,point))/(self.sigma**2)
        return sum
    
    def complete_gradient(self,point):
        product = np.dot(self.Knm_matrix.T,self.Knm_matrix) 
        grad = np.dot(self.Kmm_matrix+(1/self.sigma**2)*product,point)
        grad -= (1/self.sigma**2)*np.dot(self.Y.T,self.Knm_matrix)
        return(grad)
    
    def partial_gradient(self,agent_index,point,agent_index_2):
        sum  = np.dot(self.Kmm_matrix,point)/5    
        matrix_Kim = np.array([kernel(self.X[self.data_point_of_agent[agent_index,agent_index_2]],self.centers_kernel[j]) for j in range(self.number_of_kernels)])
        sum -=matrix_Kim*(self.Y[self.data_point_of_agent[agent_index,agent_index_2]]-np.dot(matrix_Kim,point))/(self.sigma**2)
        return sum

class solver:
    def __init__(self, instance1, step_size, number_iteration, initialisation) -> None:
        self.instance = instance1
        self.step_size = step_size
        self.number_iteration = number_iteration
        self.initialisation = initialisation
        self.variable_zize = np.shape(initialisation)[0]
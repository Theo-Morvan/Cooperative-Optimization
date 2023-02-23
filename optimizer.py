import numpy as np

def kernel(x,y):
    return np.exp(-(x-y)**2)


class instance:
    def __init__(self,x,y,agents,adjacence_matrix,sigma) -> None:
        self.X=x
        self.Y=y
        self.agents = agents
        self.adjacence_matrix = adjacence_matrix
        self.sigma = sigma
        data_size = np.shape(x)[0]
        self.data_size = data_size
        self.Knn_matrix = np.array([[kernel(x[k],x[l]) for k in range(data_size)]for l in range(data_size)])
        number_of_agents,data_per_agent = np.shape(agents) 
        self.number_of_agents = number_of_agents
        self.data_per_agent = data_per_agent

    def objective(self,curent_solution):
        objecti = np.dot(curent_solution,np.dot(self.Knn_matrix,curent_solution))
        for agent_index in range(self.data_size):
            matrix_Kim = np.array([kernel(self.X[agent_index],self.X[agent_index_2]) for agent_index_2 in range(self.data_size)])
            objecti+=(1/(2*(self.sigma**2)))*(self.Y[agent_index]-np.dot(matrix_Kim,curent_solution))**2
        return(objecti)

class solver : 

    def __init__(self,instance1,step_size,number_iteration,initialisation) -> None:
        self.instance = instance1
        self.step_size = step_size
        self.number_iteration = number_iteration
        self.initialisation = initialisation
        self.variable_zize = np.shape(initialisation)[0]
        

class DGD(solver):
    def __init__(self,instance1,step_size,number_iteration,initialisation) -> None:
        super().__init__(instance1,step_size,number_iteration,initialisation)
        self.curent_solution = np.array([initialisation for i in range(instance1.number_of_agents)])
        self.new_solution = np.array([initialisation for i in range(instance1.number_of_agents)])
    
    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print("Agent "+str(agent_index)+" objective : "+str(self.instance.objective(self.curent_solution[agent_index])))

    def solve(self):
        for iteration in range(self.number_iteration):
            print("Iterration : "+str(iteration))
            self.do_optimisation_step_DGD()
        return(self.curent_solution)
    
    def do_optimisation_step_DGD(self):
        self.new_solution = np.zeros([self.instance.number_of_agents,self.variable_zize])
        for agent_index in range(self.instance.number_of_agents):
            self.do_local_optimisation_step_DGD(agent_index)
        self.display_objective()
        self.curent_solution = self.new_solution

    def do_local_optimisation_step_DGD(self,agent_index):
        gradient = self.gradientDGD(agent_index)
        print("Agent " + str(agent_index)+" norme du gradient : "+str(np.linalg.norm(gradient)))
        for agent_index_2 in range(self.instance.number_of_agents):
            self.new_solution[agent_index]=self.instance.adjacence_matrix[agent_index,agent_index_2]*self.curent_solution[agent_index_2]-self.step_size*gradient

    def gradientDGD(self,agent_index):
        first_term = np.dot(self.instance.Knn_matrix,self.curent_solution[agent_index])/5
        matrix_K0m =np.array([kernel(self.instance.X[self.instance.agents[agent_index,0]],self.instance.X[j]) for j in range(self.instance.data_size)])
        matrix_K1m = np.array([kernel(self.instance.X[self.instance.agents[agent_index,1]],self.instance.X[j]) for j in range(self.instance.data_size)])
        second_term = (matrix_K0m*(self.instance.Y[self.instance.agents[agent_index,0]]-np.dot(matrix_K0m,self.curent_solution[agent_index])) + 
                        matrix_K1m*(self.instance.Y[self.instance.agents[agent_index,1]]-np.dot(matrix_K1m,self.curent_solution[agent_index])))/(self.instance.sigma**2)
        return first_term - second_term



class gradient_tracking(solver):

    def __init__(self,instance1,step_size,number_iteration,initialisation) -> None:
        super().__init__(instance1,step_size,number_iteration,initialisation)

    def gradient_tracking(self):
        return


class dual_decomposition(solver):

    def __init__(self,instance1,step_size,number_iteration,initialisation) -> None:
        super().__init__(instance1,step_size,number_iteration,initialisation)

    def dual_decomposition(self):
        return

class ADMM(solver):

    def __init__(self,instance1,step_size,number_iteration,initialisation) -> None:
        super().__init__(instance1,step_size,number_iteration,initialisation)

    def ADMM(self):
        return

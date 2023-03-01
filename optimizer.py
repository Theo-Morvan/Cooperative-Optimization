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
        matrixA = np.identity(number_of_agents*data_size)
        for index in range(data_size):
                matrixA[(number_of_agents-1)*data_size+index,(number_of_agents-1)*data_size+index]=0
        for index in range((number_of_agents-1)*data_size):
                matrixA[index,index+data_size]=-1
        self.matrixA = matrixA

    def objective(self,curent_solution):
        objecti = np.dot(curent_solution,np.dot(self.Knn_matrix,curent_solution))
        for agent_index in range(self.data_size):
            matrix_Kim = np.array([kernel(self.X[agent_index],self.X[agent_index_2]) for agent_index_2 in range(self.data_size)])
            objecti+=(1/(2*(self.sigma**2)))*(self.Y[agent_index]-np.dot(matrix_Kim,curent_solution))**2
        return(objecti)
    
    def gradient(self,agent_index,point):
        first_term = np.dot(self.Knn_matrix,point)/5
        matrix_K0m =np.array([kernel(self.X[self.agents[agent_index,0]],self.X[j]) for j in range(self.data_size)])
        matrix_K1m = np.array([kernel(self.X[self.agents[agent_index,1]],self.X[j]) for j in range(self.data_size)])
        second_term = (matrix_K0m*(self.Y[self.agents[agent_index,0]]-np.dot(matrix_K0m,point)) + 
                        matrix_K1m*(self.Y[self.agents[agent_index,1]]-np.dot(matrix_K1m,point)))/(self.sigma**2)
        return first_term - second_term


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

    def solve(self):
        for iteration in range(self.number_iteration):
            print("Iterration : "+str(iteration))
            self.do_optimisation_step_DGD()
        return(self.curent_solution)
    
    def do_optimisation_step_DGD(self):
        self.new_solution = np.zeros([self.instance.number_of_agents,self.variable_zize])
        for agent_index in range(self.instance.number_of_agents):
            self.do_local_gradient_descent(agent_index)
        self.display_objective()
        self.curent_solution = self.new_solution

    def do_local_gradient_descent(self,agent_index):
        gradient = self.instance.gradient(agent_index,self.curent_solution[agent_index])
        print("Agent " + str(agent_index)+" norme du gradient : "+str(np.linalg.norm(gradient)))
        for agent_index_2 in range(self.instance.number_of_agents):
            self.new_solution[agent_index]+=self.instance.adjacence_matrix[agent_index,agent_index_2]*self.curent_solution[agent_index_2]
        self.new_solution[agent_index]-=self.step_size*gradient

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print("Agent "+str(agent_index)+" objective : "+str(self.instance.objective(self.curent_solution[agent_index])))


class gradient_tracking(DGD):

    def __init__(self,instance1,step_size,number_iteration,initialisation) -> None:
        super().__init__(instance1,step_size,number_iteration,initialisation)
        self.curent_gradient_like = np.array([self.instance.gradient(agent_index,initialisation) for agent_index in range(instance1.number_of_agents)])
        self.new_gradient_like = np.array([self.instance.gradient(agent_index,initialisation) for agent_index in range(instance1.number_of_agents)])

    def solve(self):
        for iteration in range(self.number_iteration):
            print("Iterration : "+str(iteration))
            self.do_optimisation_step_gradient_tracking()
        # print(self.curent_solution)
        return(self.curent_solution)

    def do_optimisation_step_gradient_tracking(self):
        self.new_solution = np.zeros([self.instance.number_of_agents,self.variable_zize])
        self.new_gradient_like = np.zeros([self.instance.number_of_agents,self.variable_zize])

        for agent_index in range(self.instance.number_of_agents):
            self.do_local_gradient_descent(agent_index)

        for agent_index in range(self.instance.number_of_agents):
            self.do_local_gradient_like_update(agent_index)

        # print(self.new_solution)

        self.curent_solution = self.new_solution
        self.curent_gradient_like = self.new_gradient_like
        self.display_objective()

    def do_local_gradient_descent(self,agent_index):
        gradient = self.curent_gradient_like[agent_index]
        print("Agent " + str(agent_index)+" norme du gradient (gradient like) : "+str(np.linalg.norm(gradient)))
        for agent_index_2 in range(self.instance.number_of_agents):
            self.new_solution[agent_index]+=self.instance.adjacence_matrix[agent_index,agent_index_2]*self.curent_solution[agent_index_2]
        self.new_solution[agent_index]-=self.step_size*gradient

    def do_local_gradient_like_update(self,agent_index):
        for agent_index_2 in range(self.instance.number_of_agents):
            self.new_gradient_like[agent_index]+= self.instance.adjacence_matrix[agent_index,agent_index_2]*self.curent_gradient_like[agent_index_2]
        gradient_iteration_k = self.instance.gradient(agent_index,self.curent_solution[agent_index])
        gradient_iteration_k_plus_1 = self.instance.gradient(agent_index,self.new_solution[agent_index])
        self.new_gradient_like[agent_index]-= gradient_iteration_k
        self.new_gradient_like[agent_index]+= gradient_iteration_k_plus_1
        
    def display_objective(self):
        return super().display_objective()

class dual_decomposition(solver):

    def __init__(self,instance1,step_size_primal,step_size_dual,number_iteration_primal,number_iteration_dual,initialisation, initialisation_dual) -> None:
        super().__init__(instance1,step_size_primal,number_iteration_primal,initialisation)
        self.curent_primal_solution = np.array([initialisation for i in range(instance1.number_of_agents)])
        self.new_primal_solution = np.array([initialisation for i in range(instance1.number_of_agents)])
        self.step_size_dual = step_size_dual
        self.number_iteration_dual = number_iteration_dual
        self.curent_dual_variable = initialisation_dual
        self.new_dual_variable = initialisation_dual


    def solve(self):
        for iteration in range(self.number_iteration_dual):
            print("Iteration : "+str(iteration))
            self.compute_primal_variableS()
            self.do_dual_ascent_step()
            self.curent_dual_variable = self.new_dual_variable
            self.display_objective()
        return
    
    def compute_primal_variableS(self):
        for agent_index in range(self.instance.number_of_agents):
              self.compute_primal_variable(agent_index)
        self.curent_primal_solution = self.new_primal_solution

    def compute_primal_variable(self,agent_index):
         for primal_iteration in range(self.number_iteration):
              self.do_primal_iteration_step(agent_index)

    def do_primal_iteration_step(self,agent_index):
         gradient = self.instance.gradient(agent_index,self.curent_primal_solution[agent_index]) 
         agent_index_th_column_of_A = self.instance.matrixA[:,agent_index*self.instance.data_size:(agent_index+1)*self.instance.data_size].T
         communication_vector = np.array([self.instance.adjacence_matrix[agent_index] for index in range(self.instance.data_size)]).T.flatten()
         components_of_dual_variable_communicated = self.curent_dual_variable*communication_vector
         gradient += np.dot(agent_index_th_column_of_A,components_of_dual_variable_communicated)
         self.new_primal_solution[agent_index]=self.curent_primal_solution[agent_index]
         self.new_primal_solution[agent_index]-=self.step_size*gradient
         

    def do_dual_ascent_step(self):
        self.new_dual_variable = self.curent_dual_variable
        gradient = np.dot(self.instance.matrixA,self.curent_primal_solution.flatten())
        print("Norme du gradient : "+str(np.linalg.norm(gradient)))
        self.new_dual_variable += self.step_size_dual*gradient

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print("Agent "+str(agent_index)+" objective : "+str(self.instance.objective(self.curent_primal_solution[agent_index])))


class dual_decomposition(solver):

    def __init__(self,instance1,step_size_primal,step_size_dual,number_iteration_primal,number_iteration_dual,initialisation, initialisation_dual) -> None:
        super().__init__(instance1,step_size_primal,number_iteration_primal,initialisation)
        self.curent_primal_solution = np.array([initialisation for i in range(instance1.number_of_agents)])
        self.new_primal_solution = np.array([initialisation for i in range(instance1.number_of_agents)])
        self.step_size_dual = step_size_dual
        self.number_iteration_dual = number_iteration_dual
        self.curent_dual_variable = initialisation_dual
        self.new_dual_variable = initialisation_dual


    def solve(self):
        for iteration in range(self.number_iteration_dual):
            print("Iteration : "+str(iteration))
            self.compute_primal_variableS()
            self.do_dual_ascent_step()
            self.curent_dual_variable = self.new_dual_variable
            self.display_objective()
        return
    
    def compute_primal_variableS(self):
        for agent_index in range(self.instance.number_of_agents):
              self.compute_primal_variable(agent_index)
        self.curent_primal_solution = self.new_primal_solution

    def compute_primal_variable(self,agent_index):
         for primal_iteration in range(self.number_iteration):
              self.do_primal_iteration_step(agent_index)

    def do_primal_iteration_step(self,agent_index):
         gradient = self.instance.gradient(agent_index,self.curent_primal_solution[agent_index]) 
         agent_index_th_column_of_A = self.instance.matrixA[:,agent_index*self.instance.data_size:(agent_index+1)*self.instance.data_size].T
         communication_vector = np.array([self.instance.adjacence_matrix[agent_index] for index in range(self.instance.data_size)]).T.flatten()
         components_of_dual_variable_communicated = self.curent_dual_variable*communication_vector
         gradient += np.dot(agent_index_th_column_of_A,components_of_dual_variable_communicated)
         self.new_primal_solution[agent_index]=self.curent_primal_solution[agent_index]
         self.new_primal_solution[agent_index]-=self.step_size*gradient
         

    def do_dual_ascent_step(self):
        self.new_dual_variable = self.curent_dual_variable
        gradient = np.dot(self.instance.matrixA,self.curent_primal_solution.flatten())
        print("Norme du gradient : "+str(np.linalg.norm(gradient)))
        self.new_dual_variable += self.step_size_dual*gradient

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print("Agent "+str(agent_index)+" objective : "+str(self.instance.objective(self.curent_primal_solution[agent_index])))


class dual_decomposition_edge(solver):

    def __init__(self,instance1,step_size_primal,step_size_dual,number_iteration_primal,number_iteration_dual,initialisation, initialisation_dual) -> None:
        super().__init__(instance1,step_size_primal,number_iteration_primal,initialisation)
        self.curent_primal_solution = np.array([initialisation for i in range(instance1.number_of_agents)])
        self.new_primal_solution = np.array([initialisation for i in range(instance1.number_of_agents)])
        self.step_size_dual = step_size_dual
        self.number_iteration_dual = number_iteration_dual
        #one dual variable per edge of the graph
        self.curent_dual_variable = np.array([[initialisation_dual for index in range(self.instance.number_of_agents)]for index2 in range(self.instance.number_of_agents)])
        self.new_dual_variable = np.array([[initialisation_dual for index in range(self.instance.number_of_agents)]for index2 in range(self.instance.number_of_agents)])


    def solve(self):
        for iteration in range(self.number_iteration_dual):
            print("Iteration : "+str(iteration))
            self.compute_primal_variableS()
            self.do_dual_ascent_step()
            self.curent_dual_variable = self.new_dual_variable
            self.display_objective()
        return
    
    def compute_primal_variableS(self):
        for agent_index in range(self.instance.number_of_agents):
              self.compute_primal_variable(agent_index)
        self.curent_primal_solution = self.new_primal_solution

    def compute_primal_variable(self,agent_index):
         for primal_iteration in range(self.number_iteration):
              self.do_primal_iteration_step(agent_index)

    def do_primal_iteration_step(self,agent_index):
        gradient = self.instance.gradient(agent_index,self.curent_primal_solution[agent_index]) 
        for agent_index_2 in range(self.instance.number_of_agents):
            gradient += self.instance.adjacence_matrix[agent_index,agent_index_2]*(self.curent_dual_variable[agent_index,agent_index_2]-self.curent_dual_variable[agent_index_2,agent_index])
        self.new_primal_solution[agent_index]=self.curent_primal_solution[agent_index]
        self.new_primal_solution[agent_index]-=self.step_size*gradient
    
    def do_dual_ascent_step(self):
        self.new_dual_variable = self.curent_dual_variable
        for agent_index_1 in range(self.instance.number_of_agents):
            for agent_index_2 in range(self.instance.number_of_agents):
                self.do_dual_ascent_step_on_edge(agent_index_1,agent_index_2)
    
    def do_dual_ascent_step_on_edge(self,agent_index_1,agent_index_2):
        self.new_dual_variable[agent_index_1,agent_index_2]+=self.step_size_dual*(self.curent_primal_solution[agent_index_1]-self.curent_primal_solution[agent_index_2])


    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print("Agent "+str(agent_index)+" objective : "+str(self.instance.objective(self.curent_primal_solution[agent_index])))



class ADMM(solver):

    def __init__(self,instance1,step_size,number_iteration,initialisation) -> None:
        super().__init__(instance1,step_size,number_iteration,initialisation)

    def solve(self):
        return

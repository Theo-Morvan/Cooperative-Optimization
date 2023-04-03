import numpy as np
import pandas as pd
from solver import *
import ipdb


class DGD_DP(solver):
    def __init__(self, instance1, step_size, number_iteration, initialisation) -> None:
        super().__init__(instance1, step_size, number_iteration, initialisation)
        self.current_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.new_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.distance_to_optimum = np.zeros(
            (self.number_iteration, instance1.number_of_agents)
        )
        self.original_communication_matrix = (
            compute_doubly_stochastic_matrix(
                self.instance.communication_matrix
            )
            - np.eye(instance1.number_of_agents)
        )
        self.effective_communication_matrix = (
            compute_doubly_stochastic_matrix(
                self.instance.communication_matrix
            )
            - np.eye(instance1.number_of_agents)
        )


    def decreasing_weighted_scale(self, iter, gamma_value, decay:float=0.):
        return gamma_value * 1/(1+decay*iter)
    
    def send_obscured_solution(self, laplacian_scale:float):
        current_solutions = self.current_solution
        laplacian_noise = np.random.laplace(loc=0, scale=laplacian_scale,size=current_solutions.shape)
        # try:
        #     print(current_solutions.shape)
        # except:
        #     ipdb.set_trace()
        obscured_solutions = current_solutions + laplacian_noise
        return obscured_solutions



    def update_distance_to_optimum(self, iteration):
        for agent_index in range(self.instance.number_of_agents):
            self.distance_to_optimum[iteration, agent_index] = np.linalg.norm(
                self.current_solution[agent_index] - self.instance.true_solution
            )
    def reinitialize_communication_matrix(self):
        self.effective_communication_matrix = self.original_communication_matrix
    

    def do_local_gradient_descent(self, agent_index, gamma_value:float,verbose=True):
        gradient = self.instance.gradient(
            agent_index, self.current_solution[agent_index]
        )
        if verbose:
            print(
                "Agent "
                + str(agent_index)
                + " norme du gradient : "
                + str(np.linalg.norm(gradient))
            )
        
        x_k_i = self.current_solution[agent_index]
        # self.new_solution[agent_index]+=x_k_i
        i= 0
        solution = 0
        solution += 0*x_k_i
        for agent_index_2 in range(self.instance.number_of_agents):
            x_k_j_obscured = self.obscured_solutions[agent_index_2]
            x_k_j = self.current_solution[agent_index_2]
            w_i_j = self.effective_communication_matrix[agent_index, agent_index_2]
            # noise_w_j = np.random.laplace(loc=0, scale=self.laplacian_scale, size=x_k_j.shape)
            solution += gamma_value*w_i_j *(x_k_j_obscured - 1*x_k_i)
            # weighted matrix part of update step for each agent. We need to add the laplacian noise here
        solution -= self.step_size * gradient
        self.new_solution[agent_index] = solution
        # ipdb.set_trace()


    def do_optimisation_step_DGD_DP(
        self, gamma_value:float,verbose=True, package_loss=False, probability_package_loss=0
    ):
        self.new_solution = np.zeros(
            [self.instance.number_of_agents, self.variable_zize]
        )

        for agent_index in range(self.instance.number_of_agents):
            self.do_local_gradient_descent(agent_index, gamma_value,verbose)

        if verbose:
            self.display_objective()
        self.current_solution = self.new_solution
        self.obscured_solutions = self.send_obscured_solution(self.laplacian_scale)

    def sample_effective_communication_matrix(self, probability_package_loss):
        sampled_adjacence_matrix = np.copy(self.instance.communication_matrix)
        n, p = np.shape(self.instance.communication_matrix)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if sampled_adjacence_matrix[i, j] != 0:
                    keep_arc_i_j = np.random.binomial(1, 1 - probability_package_loss)
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
        self.effective_communication_matrix = compute_doubly_stochastic_matrix(
            sampled_adjacence_matrix
        )


    
    def solve(self, step_size:float, laplacian_scale:float=1.,gamma_init:float=1.,verbose=True, package_loss=False, probability_package_loss=0):
        self.laplacian_scale = laplacian_scale
        self.gamma_value = gamma_init
        self.obscured_solutions = self.send_obscured_solution(self.laplacian_scale)
        self.init_step_size = step_size
        for iteration in range(self.number_iteration):
            self.step_size = self.init_step_size/(1+iteration)
            self.gamma_value = 1/((1+iteration)**(0.9))
            self.laplacian_scale = ((1+iteration)**(0.3))
            if verbose:
                print("Iterration : " + str(iteration))
            if package_loss:
                self.sample_effective_communication_matrix(probability_package_loss)

            self.do_optimisation_step_DGD_DP(
                iteration, verbose, package_loss, probability_package_loss
            )

            if package_loss:
                self.reinitialize_communication_matrix()

            self.update_distance_to_optimum(iteration)
            # self.gamma_value = self.decreasing_weighted_scale(iteration, self.gamma_value, decay)
            # print(self.gamma_value)
            if np.isnan(np.min(self.current_solution)):
                ipdb.set_trace()
        return self.current_solution

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print(
                "Agent "
                + str(agent_index + 1)
                + " objective : "
                + str(self.instance.objective(self.current_solution[agent_index]))
            )

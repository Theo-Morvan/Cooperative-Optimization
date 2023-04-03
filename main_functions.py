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
    def __init__(
        self,
        X,
        centers_kernel,
        Y,
        kernel_center_of_agent,
        data_point_of_agent,
        communication_matrix,
        sigma,
    ) -> None:
        self.centers_kernel = centers_kernel
        self.X = X
        self.Y = Y
        self.kernel_center_of_agent = kernel_center_of_agent
        self.data_point_of_agent = data_point_of_agent
        self.communication_matrix = communication_matrix
        self.sigma = sigma
        number_of_kernels = np.shape(centers_kernel)[0]
        number_of_datas = np.shape(X)[0]
        self.number_of_datas = number_of_datas
        self.number_of_kernels = number_of_kernels
        self.Kmm_matrix = np.array(
            [
                [
                    kernel(centers_kernel[k], centers_kernel[l])
                    for k in range(number_of_kernels)
                ]
                for l in range(number_of_kernels)
            ]
        )
        self.Knm_matrix = np.array(
            [
                [kernel(centers_kernel[k], X[l]) for k in range(number_of_kernels)]
                for l in range(self.number_of_datas)
            ]
        )
        number_of_agents, kernels_per_agent = np.shape(kernel_center_of_agent)
        _, data_per_agent = np.shape(data_point_of_agent)
        self.number_of_agents = number_of_agents
        self.kernels_per_agent = kernels_per_agent
        self.data_per_agent = data_per_agent
        matrixA = np.identity(number_of_agents * number_of_kernels)
        for index in range(number_of_kernels):
            matrixA[
                (number_of_agents - 1) * number_of_kernels + index,
                (number_of_agents - 1) * number_of_kernels + index,
            ] = 0
        for index in range((number_of_agents - 1) * number_of_kernels):
            matrixA[index, index + number_of_kernels] = -1
        self.matrixA = matrixA
        self.true_solution = np.zeros(number_of_agents)

    def objective(self, curent_solution):
        objecti = np.dot(curent_solution, np.dot(self.Kmm_matrix, curent_solution)) / 2
        for agent_index in range(self.number_of_agents):
            for agent_index_2 in range(self.data_per_agent):
                matrix_Kim = np.array(
                    [
                        kernel(
                            self.X[
                                self.data_point_of_agent[agent_index, agent_index_2]
                            ],
                            self.centers_kernel[j],
                        )
                        for j in range(self.number_of_kernels)
                    ]
                )
                objecti += (1 / (2 * (self.sigma**2))) * (
                    self.Y[self.data_point_of_agent[agent_index, agent_index_2]]
                    - np.dot(matrix_Kim, curent_solution)
                ) ** 2
        return objecti

    def gradient(self, agent_index, point):
        sum = np.dot(self.Kmm_matrix, point) / 5
        for agent_index_2 in range(self.data_per_agent):
            matrix_Kim = np.array(
                [
                    kernel(
                        self.X[self.data_point_of_agent[agent_index, agent_index_2]],
                        self.centers_kernel[j],
                    )
                    for j in range(self.number_of_kernels)
                ]
            )
            sum -= (
                matrix_Kim
                * (
                    self.Y[self.data_point_of_agent[agent_index, agent_index_2]]
                    - np.dot(matrix_Kim, point)
                )
                / (self.sigma**2)
            )
        return sum

    def complete_gradient(self, point):
        product = np.dot(self.Knm_matrix.T, self.Knm_matrix)
        grad = np.dot(self.Kmm_matrix + (1 / self.sigma**2) * product, point)
        grad -= (1 / self.sigma**2) * np.dot(self.Y.T, self.Knm_matrix)
        return grad

    def partial_gradient(self, agent_index, point, agent_index_2):
        sum = np.dot(self.Kmm_matrix, point) / 5
        matrix_Kim = np.array(
            [
                kernel(
                    self.X[self.data_point_of_agent[agent_index, agent_index_2]],
                    self.centers_kernel[j],
                )
                for j in range(self.number_of_kernels)
            ]
        )
        sum -= (
            matrix_Kim
            * (
                self.Y[self.data_point_of_agent[agent_index, agent_index_2]]
                - np.dot(matrix_Kim, point)
            )
            / (self.sigma**2)
        )
        return sum


class solver:
    def __init__(self, instance1, step_size, number_iteration, initialisation) -> None:
        self.instance = instance1
        self.step_size = step_size
        self.number_iteration = number_iteration
        self.initialisation = initialisation
        self.variable_zize = np.shape(initialisation)[0]


class gradient_descent(solver):
    def __init__(self, instance1, step_size, number_iteration, initialisation) -> None:
        super().__init__(instance1, step_size, number_iteration, initialisation)
        self.curent_solution = initialisation
        self.new_solution = initialisation

    def solve(self, verbose=True):
        for iteration in range(self.number_iteration):
            self.do_gradient_descent_step()
            self.curent_solution = self.new_solution
            if verbose:
                self.display_objective()
        return self.curent_solution

    def do_gradient_descent_step(self):
        gradient = self.instance.complete_gradient(self.curent_solution)
        self.new_solution = self.curent_solution - self.step_size * gradient

    def display_objective(self):
        print(
            "Objective : " + str(self.instance.objective(self.curent_server_solution))
        )


class DGD(solver):
    def __init__(self, instance1, step_size, number_iteration, initialisation) -> None:
        super().__init__(instance1, step_size, number_iteration, initialisation)
        self.curent_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.new_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.distance_to_optimum = np.zeros(
            (self.number_iteration, instance1.number_of_agents)
        )
        self.original_communication_matrix = compute_doubly_stochastic_matrix(
            self.instance.communication_matrix
        )
        self.effective_communication_matrix = compute_doubly_stochastic_matrix(
            self.instance.communication_matrix
        )

    def solve(self, verbose=True, package_loss=False, probability_package_loss=0):
        for iteration in range(self.number_iteration):
            if verbose:
                print("Iterration : " + str(iteration))
            if package_loss:
                self.sample_effective_communication_matrix(probability_package_loss)

            self.do_optimisation_step_DGD(
                verbose, package_loss, probability_package_loss
            )

            if package_loss:
                self.reinitialize_communication_matrix()

            self.update_distance_to_optimum(iteration)
        return self.curent_solution

    def update_distance_to_optimum(self, iteration):
        for agent_index in range(self.instance.number_of_agents):
            self.distance_to_optimum[iteration, agent_index] = np.linalg.norm(
                self.curent_solution[agent_index] - self.instance.true_solution
            )

    def do_optimisation_step_DGD(
        self, verbose=True, package_loss=False, probability_package_loss=0
    ):
        self.new_solution = np.zeros(
            [self.instance.number_of_agents, self.variable_zize]
        )

        for agent_index in range(self.instance.number_of_agents):
            self.do_local_gradient_descent(agent_index, verbose)

        if verbose:
            self.display_objective()
        self.curent_solution = self.new_solution

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

    def reinitialize_communication_matrix(self):
        self.effective_communication_matrix = self.original_communication_matrix

    def do_local_gradient_descent(self, agent_index, verbose=True):
        gradient = self.instance.gradient(
            agent_index, self.curent_solution[agent_index]
        )
        if verbose:
            print(
                "Agent "
                + str(agent_index)
                + " norme du gradient : "
                + str(np.linalg.norm(gradient))
            )
        for agent_index_2 in range(self.instance.number_of_agents):
            self.new_solution[agent_index] += (
                self.effective_communication_matrix[agent_index, agent_index_2]
                * self.curent_solution[agent_index_2]
            )  # weighted matrix part of update step for each agent. We need to add the laplacian noise here
        self.new_solution[agent_index] -= self.step_size * gradient

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print(
                "Agent "
                + str(agent_index + 1)
                + " objective : "
                + str(self.instance.objective(self.curent_solution[agent_index]))
            )


class gradient_tracking(DGD):
    def __init__(self, instance1, step_size, number_iteration, initialisation) -> None:
        super().__init__(instance1, step_size, number_iteration, initialisation)
        self.curent_gradient_like = np.array(
            [
                self.instance.gradient(agent_index, initialisation)
                for agent_index in range(instance1.number_of_agents)
            ]
        )
        self.new_gradient_like = np.array(
            [
                self.instance.gradient(agent_index, initialisation)
                for agent_index in range(instance1.number_of_agents)
            ]
        )

    def reinitialize_communication_matrix(self):
        return super().reinitialize_communication_matrix()

    def sample_effective_communication_matrix(self, probability_package_loss):
        return super().sample_effective_communication_matrix(probability_package_loss)

    def solve(self, verbose=True, package_loss=False, probability_package_loss=0):
        for iteration in range(self.number_iteration):
            if verbose:
                print("Iterration : " + str(iteration))
            self.do_optimisation_step_gradient_tracking(
                verbose, package_loss, probability_package_loss
            )
            self.update_distance_to_optimum(iteration)
        return self.curent_solution

    def do_optimisation_step_gradient_tracking(
        self, verbose=True, package_loss=False, probability_package_loss=0
    ):
        self.new_solution = np.zeros(
            [self.instance.number_of_agents, self.variable_zize]
        )
        self.new_gradient_like = np.zeros(
            [self.instance.number_of_agents, self.variable_zize]
        )
        if package_loss:
            self.sample_effective_communication_matrix(probability_package_loss)

        for agent_index in range(self.instance.number_of_agents):
            self.do_local_gradient_descent(agent_index, verbose)

        for agent_index in range(self.instance.number_of_agents):
            self.do_local_gradient_like_update(agent_index)
        if package_loss:
            self.reinitialize_communication_matrix()

        self.curent_solution = self.new_solution
        self.curent_gradient_like = self.new_gradient_like
        if verbose:
            self.display_objective()

    def do_local_gradient_descent(self, agent_index, verbose=True):
        gradient = self.curent_gradient_like[agent_index]
        if verbose:
            print(
                "Agent "
                + str(agent_index)
                + " norme du gradient (gradient like) : "
                + str(np.linalg.norm(gradient))
            )
        for agent_index_2 in range(self.instance.number_of_agents):
            self.new_solution[agent_index] += (
                self.effective_communication_matrix[agent_index, agent_index_2]
                * self.curent_solution[agent_index_2]
            )
        self.new_solution[agent_index] -= self.step_size * gradient

    def do_local_gradient_like_update(self, agent_index):
        for agent_index_2 in range(self.instance.number_of_agents):
            self.new_gradient_like[agent_index] += (
                self.effective_communication_matrix[agent_index, agent_index_2]
                * self.curent_gradient_like[agent_index_2]
            )
        gradient_iteration_k = self.instance.gradient(
            agent_index, self.curent_solution[agent_index]
        )
        gradient_iteration_k_plus_1 = self.instance.gradient(
            agent_index, self.new_solution[agent_index]
        )
        self.new_gradient_like[agent_index] -= gradient_iteration_k
        self.new_gradient_like[agent_index] += gradient_iteration_k_plus_1

    def display_objective(self):
        return super().display_objective()

    def update_distance_to_optimum(self, iteration):
        return super().update_distance_to_optimum(iteration)


class dual_decomposition(solver):
    def __init__(
        self,
        instance1,
        step_size_primal,
        step_size_dual,
        number_iteration_primal,
        number_iteration_dual,
        initialisation,
        initialisation_dual,
    ) -> None:
        super().__init__(
            instance1, step_size_primal, number_iteration_primal, initialisation
        )
        self.curent_primal_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.new_primal_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.step_size_dual = step_size_dual
        self.number_iteration_dual = number_iteration_dual
        self.curent_dual_variable = initialisation_dual
        self.new_dual_variable = initialisation_dual
        self.distance_to_optimum = np.zeros(
            (self.number_iteration_dual, instance1.number_of_agents)
        )
        self.original_communication_matrix = self.instance.communication_matrix
        self.effective_communication_matrix = self.instance.communication_matrix

    def sample_effective_communication_matrix(self, probability_package_loss):
        sampled_adjacence_matrix = np.copy(self.instance.communication_matrix)
        n, p = np.shape(self.instance.communication_matrix)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if sampled_adjacence_matrix[i, j] != 0:
                    keep_arc_i_j = np.random.binomial(1, 1 - probability_package_loss)
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
        self.effective_communication_matrix = sampled_adjacence_matrix

    def reinitialize_communication_matrix(self):
        self.effective_communication_matrix = self.original_communication_matrix

    def solve(self, verbose=True, package_loss=False, probability_package_loss=0):
        for iteration in range(self.number_iteration_dual):
            if verbose:
                print("Iteration : " + str(iteration))
            if package_loss:
                self.sample_effective_communication_matrix(probability_package_loss)
            self.compute_primal_variableS()
            self.do_dual_ascent_step(verbose)
            self.curent_dual_variable = self.new_dual_variable
            if package_loss:
                self.reinitialize_communication_matrix()
            if verbose:
                self.display_objective()
            self.update_distance_to_optimum(iteration)
        return

    def compute_primal_variableS(self):
        for agent_index in range(self.instance.number_of_agents):
            self.compute_primal_variable(agent_index)
        self.curent_primal_solution = self.new_primal_solution

    def compute_primal_variable(self, agent_index):
        for primal_iteration in range(self.number_iteration):
            self.do_primal_iteration_step(agent_index)

    def do_primal_iteration_step(self, agent_index):
        gradient = self.instance.gradient(
            agent_index, self.curent_primal_solution[agent_index]
        )
        agent_index_th_column_of_A = self.instance.matrixA[
            :,
            agent_index
            * self.instance.number_of_kernels : (agent_index + 1)
            * self.instance.number_of_kernels,
        ].T
        communication_vector = np.array(
            [
                self.effective_communication_matrix[agent_index]
                for index in range(self.instance.number_of_kernels)
            ]
        ).T.flatten()
        components_of_dual_variable_communicated = (
            self.curent_dual_variable * communication_vector
        )
        gradient += np.dot(
            agent_index_th_column_of_A, components_of_dual_variable_communicated
        )
        self.new_primal_solution[agent_index] = self.curent_primal_solution[agent_index]
        self.new_primal_solution[agent_index] -= self.step_size * gradient

    def do_dual_ascent_step(self, verbose=True):
        self.new_dual_variable = self.curent_dual_variable
        gradient = np.dot(self.instance.matrixA, self.curent_primal_solution.flatten())
        if verbose:
            print("Norme du gradient : " + str(np.linalg.norm(gradient)))
        self.new_dual_variable += self.step_size_dual * gradient

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print(
                "Agent "
                + str(agent_index + 1)
                + " objective : "
                + str(self.instance.objective(self.curent_primal_solution[agent_index]))
            )

    def update_distance_to_optimum(self, iteration):
        for agent_index in range(self.instance.number_of_agents):
            self.distance_to_optimum[iteration, agent_index] = np.linalg.norm(
                self.curent_primal_solution[agent_index] - self.instance.true_solution
            )


class dual_decomposition_edge(solver):
    def __init__(
        self,
        instance1,
        step_size_primal,
        step_size_dual,
        number_iteration_primal,
        number_iteration_dual,
        initialisation,
        initialisation_dual,
    ) -> None:
        super().__init__(
            instance1, step_size_primal, number_iteration_primal, initialisation
        )
        self.curent_primal_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.new_primal_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.step_size_dual = step_size_dual
        self.number_iteration_dual = number_iteration_dual
        self.curent_dual_variable = np.array(
            [
                [initialisation_dual for index in range(self.instance.number_of_agents)]
                for index2 in range(self.instance.number_of_agents)
            ]
        )
        self.new_dual_variable = np.array(
            [
                [initialisation_dual for index in range(self.instance.number_of_agents)]
                for index2 in range(self.instance.number_of_agents)
            ]
        )
        self.distance_to_optimum = np.zeros(
            (self.number_iteration_dual, instance1.number_of_agents)
        )
        self.original_communication_matrix = self.instance.communication_matrix
        self.effective_communication_matrix = self.instance.communication_matrix

    def sample_effective_communication_matrix(self, probability_package_loss):
        sampled_adjacence_matrix = np.copy(self.instance.communication_matrix)
        n, p = np.shape(self.instance.communication_matrix)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if sampled_adjacence_matrix[i, j] != 0:
                    keep_arc_i_j = np.random.binomial(1, 1 - probability_package_loss)
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
        self.effective_communication_matrix = sampled_adjacence_matrix

    def reinitialize_communication_matrix(self):
        self.effective_communication_matrix = self.original_communication_matrix

    def solve(self, verbose=True, package_loss=False, probability_package_loss=0):
        for iteration in range(self.number_iteration_dual):
            if verbose:
                print("Iteration : " + str(iteration))
            if package_loss:
                self.sample_effective_communication_matrix(probability_package_loss)
            self.compute_primal_variableS()
            self.do_dual_ascent_step()
            self.curent_dual_variable = self.new_dual_variable
            if package_loss:
                self.reinitialize_communication_matrix()
            if verbose:
                self.display_objective()
            self.update_distance_to_optimum(iteration)
        return

    def compute_primal_variableS(self):
        for agent_index in range(self.instance.number_of_agents):
            self.compute_primal_variable(agent_index)
        self.curent_primal_solution = self.new_primal_solution

    def compute_primal_variable(self, agent_index):
        for primal_iteration in range(self.number_iteration):
            self.do_primal_iteration_step(agent_index)

    def do_primal_iteration_step(self, agent_index):
        gradient = self.instance.gradient(
            agent_index, self.curent_primal_solution[agent_index]
        )
        for agent_index_2 in range(self.instance.number_of_agents):
            gradient += self.effective_communication_matrix[
                agent_index, agent_index_2
            ] * (
                self.curent_dual_variable[agent_index, agent_index_2]
                - self.curent_dual_variable[agent_index_2, agent_index]
            )
        self.new_primal_solution[agent_index] = self.curent_primal_solution[agent_index]
        self.new_primal_solution[agent_index] -= self.step_size * gradient

    def do_dual_ascent_step(self):
        self.new_dual_variable = self.curent_dual_variable
        for agent_index_1 in range(self.instance.number_of_agents):
            for agent_index_2 in range(self.instance.number_of_agents):
                if (
                    self.effective_communication_matrix[agent_index_1, agent_index_2]
                    == 1
                ):
                    self.do_dual_ascent_step_on_edge(agent_index_1, agent_index_2)

    def do_dual_ascent_step_on_edge(self, agent_index_1, agent_index_2):
        self.new_dual_variable[agent_index_1, agent_index_2] += self.step_size_dual * (
            self.curent_primal_solution[agent_index_1]
            - self.curent_primal_solution[agent_index_2]
        )

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print(
                "Agent "
                + str(agent_index + 1)
                + " objective : "
                + str(self.instance.objective(self.curent_primal_solution[agent_index]))
            )

    def update_distance_to_optimum(self, iteration):
        for agent_index in range(self.instance.number_of_agents):
            self.distance_to_optimum[iteration, agent_index] = np.linalg.norm(
                self.curent_primal_solution[agent_index] - self.instance.true_solution
            )


class dual_decomposition_edge_linalg(solver):
    def __init__(
        self,
        instance1,
        step_size_primal,
        step_size_dual,
        number_iteration_primal,
        number_iteration_dual,
        initialisation,
        initialisation_dual,
        regularization,
    ) -> None:
        super().__init__(
            instance1, step_size_primal, number_iteration_primal, initialisation
        )
        self.curent_primal_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.new_primal_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.step_size_dual = step_size_dual
        self.number_iteration_dual = number_iteration_dual
        self.curent_dual_variable = np.array(
            [
                [initialisation_dual for index in range(self.instance.number_of_agents)]
                for index2 in range(self.instance.number_of_agents)
            ]
        )
        self.new_dual_variable = np.array(
            [
                [initialisation_dual for index in range(self.instance.number_of_agents)]
                for index2 in range(self.instance.number_of_agents)
            ]
        )
        self.distance_to_optimum = np.zeros(
            (self.number_iteration_dual, instance1.number_of_agents)
        )
        self.original_communication_matrix = self.instance.communication_matrix
        self.effective_communication_matrix = self.instance.communication_matrix
        self.regularization = regularization

    def sample_effective_communication_matrix(self, probability_package_loss):
        sampled_adjacence_matrix = np.copy(self.instance.communication_matrix)
        n, p = np.shape(self.instance.communication_matrix)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if sampled_adjacence_matrix[i, j] != 0:
                    keep_arc_i_j = np.random.binomial(1, 1 - probability_package_loss)
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
        self.effective_communication_matrix = sampled_adjacence_matrix

    def reinitialize_communication_matrix(self):
        self.effective_communication_matrix = self.original_communication_matrix

    def solve(self, verbose=True, package_loss=False, probability_package_loss=0):
        for iteration in range(self.number_iteration_dual):
            if verbose:
                print("Iteration : " + str(iteration))
            if package_loss:
                self.sample_effective_communication_matrix(probability_package_loss)
            self.compute_primal_variableS()
            self.do_dual_ascent_step()
            self.curent_dual_variable = self.new_dual_variable
            if package_loss:
                self.reinitialize_communication_matrix()
            if verbose:
                self.display_objective()
            self.update_distance_to_optimum(iteration)
        return

    def compute_primal_variableS(self):
        for agent_index in range(self.instance.number_of_agents):
            self.compute_primal_variable(agent_index)
        self.curent_primal_solution = self.new_primal_solution

    def compute_primal_variable(self, agent_index):
        matrix_K_agent_m = self.instance.Knm_matrix[
            agent_index
            * self.instance.data_per_agent : (agent_index + 1)
            * self.instance.data_per_agent,
            :,
        ]
        regularization_matrix = self.regularization * np.identity(
            self.instance.number_of_kernels
        )
        left_matrix = (self.instance.Kmm_matrix + regularization_matrix) / 5 - (
            1 / self.instance.sigma**2
        ) * np.dot(matrix_K_agent_m.T, matrix_K_agent_m)
        y_agent = self.instance.Y[
            agent_index
            * self.instance.data_per_agent : (agent_index + 1)
            * self.instance.data_per_agent
        ]
        right_vecor = (1 / self.instance.sigma**2) * np.dot(
            matrix_K_agent_m.T, y_agent
        )
        for agent_index_2 in range(self.instance.number_of_agents):
            right_vecor += self.effective_communication_matrix[
                agent_index, agent_index_2
            ] * (
                self.curent_dual_variable[agent_index, agent_index_2]
                - self.curent_dual_variable[agent_index_2, agent_index]
            )
        # Q,R = np.linalg.qr(left_matrix)
        # solution_of_linear_system = np.linalg.solve(R,np.dot(Q.T,right_vecor))
        solution_of_linear_system = np.linalg.solve(left_matrix, right_vecor)
        self.new_primal_solution[agent_index] = solution_of_linear_system

    def do_dual_ascent_step(self):
        self.new_dual_variable = self.curent_dual_variable
        for agent_index_1 in range(self.instance.number_of_agents):
            for agent_index_2 in range(self.instance.number_of_agents):
                if (
                    self.effective_communication_matrix[agent_index_1, agent_index_2]
                    == 1
                ):
                    self.do_dual_ascent_step_on_edge(agent_index_1, agent_index_2)

    def do_dual_ascent_step_on_edge(self, agent_index_1, agent_index_2):
        self.new_dual_variable[agent_index_1, agent_index_2] += self.step_size_dual * (
            self.curent_primal_solution[agent_index_1]
            - self.curent_primal_solution[agent_index_2]
        )

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print(
                "Agent "
                + str(agent_index + 1)
                + " objective : "
                + str(self.instance.objective(self.curent_primal_solution[agent_index]))
            )

    def update_distance_to_optimum(self, iteration):
        for agent_index in range(self.instance.number_of_agents):
            self.distance_to_optimum[iteration, agent_index] = np.linalg.norm(
                self.curent_primal_solution[agent_index] - self.instance.true_solution
            )


class ADMM(solver):
    def __init__(
        self,
        instance1,
        step_size,
        number_iteration,
        initialisation,
        initialisation_y,
        beta,
    ) -> None:
        super().__init__(instance1, step_size, number_iteration, initialisation)
        self.curent_solution = np.array(
            [initialisation for index in range(self.instance.number_of_agents)]
        )
        self.new_solution = np.array(
            [initialisation for index in range(self.instance.number_of_agents)]
        )
        self.curent_y = np.array(
            [
                [initialisation_y for index in range(self.instance.number_of_agents)]
                for index_2 in range(self.instance.number_of_agents)
            ]
        )
        self.new_y = np.array(
            [
                [initialisation_y for index in range(self.instance.number_of_agents)]
                for index_2 in range(self.instance.number_of_agents)
            ]
        )
        self.beta = beta
        self.distance_to_optimum = np.zeros(
            (self.number_iteration, instance1.number_of_agents)
        )
        self.original_communication_matrix = self.instance.communication_matrix
        self.effective_communication_matrix = self.instance.communication_matrix

    def sample_effective_communication_matrix(self, probability_package_loss):
        sampled_adjacence_matrix = np.copy(self.instance.communication_matrix)
        n, p = np.shape(self.instance.communication_matrix)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if sampled_adjacence_matrix[i, j] != 0:
                    keep_arc_i_j = np.random.binomial(1, 1 - probability_package_loss)
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
                    sampled_adjacence_matrix[i, j] = keep_arc_i_j
        self.effective_communication_matrix = sampled_adjacence_matrix

    def reinitialize_communication_matrix(self):
        self.effective_communication_matrix = self.original_communication_matrix

    def solve(self, verbose=True, package_loss=False, probability_package_loss=0):
        for iteration in range(self.number_iteration):
            if verbose:
                print("Iteration : " + str(iteration))
            if package_loss:
                self.sample_effective_communication_matrix(probability_package_loss)
            self.compute_primal_variableS()
            self.curent_solution = self.new_solution
            self.update_y()
            self.curent_y = self.new_y
            if package_loss:
                self.reinitialize_communication_matrix()
            if verbose:
                self.display_objective()
            self.update_distance_to_optimum(iteration)

    def compute_primal_variableS(self):
        for agent_index in range(self.instance.number_of_agents):
            self.compute_primal_variable(agent_index)

    def compute_primal_variable(self, agent_index):
        gradient = self.instance.gradient(
            agent_index, self.curent_solution[agent_index]
        )
        for agent_index_2 in range(self.instance.number_of_agents):
            gradient += (
                self.beta
                * self.effective_communication_matrix[agent_index, agent_index_2]
                * (
                    self.curent_solution[agent_index]
                    - self.curent_y[agent_index, agent_index_2]
                )
            )
        self.new_solution[agent_index] = (
            self.curent_solution[agent_index] - self.step_size * gradient
        )

    def update_y(self):
        for agent_index in range(self.instance.number_of_agents):
            for agent_index_2 in range(self.instance.number_of_agents):
                if self.effective_communication_matrix[agent_index, agent_index_2] == 1:
                    self.do_local_update_y(agent_index, agent_index_2)

    def do_local_update_y(self, agent_index, agent_index_2):
        self.new_y[agent_index, agent_index_2] = (
            self.curent_solution[agent_index] + self.curent_solution[agent_index_2]
        ) / 2

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print(
                "Agent "
                + str(agent_index + 1)
                + " objective : "
                + str(self.instance.objective(self.curent_solution[agent_index]))
            )

    def update_distance_to_optimum(self, iteration):
        for agent_index in range(self.instance.number_of_agents):
            self.distance_to_optimum[iteration, agent_index] = np.linalg.norm(
                self.curent_solution[agent_index] - self.instance.true_solution
            )


class solver_FedAvg(solver):
    def __init__(
        self,
        instance1,
        step_size,
        number_iteration,
        initialisation,
        number_of_clients_selected,
        number_of_epochs,
        number_of_batches,
    ) -> None:
        super().__init__(instance1, step_size, number_iteration, initialisation)
        self.curent_server_solution = initialisation
        self.new_server_solution = initialisation
        self.curent_client_solution = np.array(
            [initialisation for i in range(self.instance.number_of_agents)]
        )
        self.new_client_solution = np.array(
            [initialisation for i in range(self.instance.number_of_agents)]
        )
        self.number_of_clients_per_epoch = number_of_clients_selected
        self.curent_clients = np.zeros((number_of_clients_selected))
        self.number_of_epochs = number_of_epochs
        self.number_of_batches = number_of_batches
        assert (
            self.instance.data_per_agent % self.number_of_batches == 0
        ), "number_of_batches must divide data_par_agent"
        self.batch_size = self.instance.data_per_agent // self.number_of_batches
        self.distance_to_optimum = np.zeros((self.number_iteration))
        self.distance_to_optimum_agent = np.zeros(
            (self.number_iteration, self.instance.number_of_agents)
        )
        self.server_objective = np.zeros((self.number_iteration))
        self.client_objective = np.zeros(
            (self.number_iteration, self.instance.number_of_agents)
        )

    def update_distance_to_optimum(self, iteration):
        self.distance_to_optimum[iteration] = np.linalg.norm(
            self.curent_server_solution - self.instance.true_solution
        )
        self.server_objective[iteration] = self.instance.objective(
            self.curent_server_solution
        )
        for agent_index in range(self.instance.number_of_agents):
            self.distance_to_optimum_agent[iteration, agent_index] = np.linalg.norm(
                self.curent_client_solution[agent_index] - self.instance.true_solution
            )
            self.client_objective[iteration, agent_index] = self.instance.objective(
                self.curent_client_solution[agent_index]
            )

    def display_objective(self):
        print(
            "Server objective : "
            + str(self.instance.objective(self.curent_server_solution))
        )

    def solve(self, verbose):
        for iteration in range(self.number_iteration):
            self.do_optimization_step()
            self.curent_server_solution = self.new_server_solution
            self.update_distance_to_optimum(iteration)
            if verbose:
                self.display_objective()

    def do_optimization_step(self):
        self.select_clients()
        self.do_mixing()
        for client in range(self.number_of_clients_per_epoch):
            self.do_client_update(client)

    def do_mixing(self):
        selected_client_solution = np.array(
            [
                self.curent_client_solution[client_index]
                for client_index in self.curent_clients
            ]
        )
        self.new_server_solution = np.mean(selected_client_solution, axis=0)

    def do_client_update(self, client):
        for epoch in range(self.number_of_epochs):
            batch = (self.number_of_batches * epoch) // self.number_of_epochs
            self.do_client_update_step(client, batch)
            self.curent_client_solution[client] = self.new_client_solution[client]

    def do_client_update_step(self, client, batch):
        sampled_data = batch * self.batch_size + np.random.randint(self.batch_size)
        sampled_gradient = self.instance.partial_gradient(
            client, self.curent_client_solution[client], sampled_data
        )
        self.new_client_solution[client] = (
            self.curent_client_solution[client] - self.step_size * sampled_gradient
        )

    def select_clients(self):
        self.curent_clients = np.random.choice(
            range(0, self.instance.number_of_agents),
            size=self.number_of_clients_per_epoch,
            replace=False,
        )


class DGD_DP(solver):
    def __init__(self, instance1, step_size, number_iteration, initialisation) -> None:
        super().__init__(instance1, step_size, number_iteration, initialisation)
        self.curent_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.new_solution = np.array(
            [initialisation for i in range(instance1.number_of_agents)]
        )
        self.distance_to_optimum = np.zeros(
            (self.number_iteration, instance1.number_of_agents)
        )
        self.original_communication_matrix = compute_doubly_stochastic_matrix(
            self.instance.communication_matrix
        )
        self.effective_communication_matrix = compute_doubly_stochastic_matrix(
            self.instance.communication_matrix
        )

    def solve(self, laplacian_scale:float=1,verbose=True, package_loss=False, probability_package_loss=0):
        self.laplacian_scale = laplacian_scale
        for iteration in range(self.number_iteration):
            if verbose:
                print("Iterration : " + str(iteration))
            if package_loss:
                self.sample_effective_communication_matrix(probability_package_loss)

            self.do_optimisation_step_DGD_DP(
                verbose, package_loss, probability_package_loss
            )

            if package_loss:
                self.reinitialize_communication_matrix()

            self.update_distance_to_optimum(iteration)
        return self.curent_solution

    def update_distance_to_optimum(self, iteration):
        for agent_index in range(self.instance.number_of_agents):
            self.distance_to_optimum[iteration, agent_index] = np.linalg.norm(
                self.curent_solution[agent_index] - self.instance.true_solution
            )

    def do_optimisation_step_DGD_DP(
        self, verbose=True, package_loss=False, probability_package_loss=0
    ):
        self.new_solution = np.zeros(
            [self.instance.number_of_agents, self.variable_zize]
        )

        for agent_index in range(self.instance.number_of_agents):
            self.do_local_gradient_descent(agent_index, verbose)

        if verbose:
            self.display_objective()
        self.curent_solution = self.new_solution

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

    def reinitialize_communication_matrix(self):
        self.effective_communication_matrix = self.original_communication_matrix
    

    def do_local_gradient_descent(self, agent_index, verbose=True):
        gradient = self.instance.gradient(
            agent_index, self.curent_solution[agent_index]
        )
        if verbose:
            print(
                "Agent "
                + str(agent_index)
                + " norme du gradient : "
                + str(np.linalg.norm(gradient))
            )
        
        x_k_i = self.new_solution[agent_index]

        for agent_index_2 in range(self.instance.number_of_agents):
            x_k_j = self.curent_solution[agent_index_2]
            w_i_j = self.effective_communication_matrix[agent_index, agent_index_2]
            self.new_solution[agent_index] += w_i_j * x_k_j
            # weighted matrix part of update step for each agent. We need to add the laplacian noise here
        self.new_solution[agent_index] -= self.step_size * gradient

    def display_objective(self):
        for agent_index in range(self.instance.number_of_agents):
            print(
                "Agent "
                + str(agent_index + 1)
                + " objective : "
                + str(self.instance.objective(self.curent_solution[agent_index]))
            )

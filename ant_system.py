import numpy as np
import time


def ant_system_tsp(dist, ants=10, max_iter=100, rho=0.5, alpha=1, beta=5, q0=0.9):
    # Inicialização dos feromônios
    n = dist.shape[0]
    tau = np.ones((n, n))

    # Loop principal do algoritmo
    best_cost = np.inf
    best_solution = None
    for it in range(max_iter):
        # Inicialização das formigas
        solutions = []
        for ant in range(ants):
            solution = [np.random.randint(n)]
            visited = set(solution)
            for _ in range(n-1):
                p = tau[solution[-1], :] ** alpha * \
                    (1.0 / (dist[solution[-1], :] + 1e-10)) ** beta
                p[list(visited)] = 0.0
                if np.random.rand() < q0:
                    next_node = np.argmax(p)
                else:
                    p = p / p.sum()
                    next_node = np.random.choice(range(n), p=p)
                solution.append(next_node)
                visited.add(next_node)
            solutions.append(solution)

        # Atualização dos feromônios
        costs = np.array([sum(dist[solution[i], solution[i+1]] for i in range(n-1)) +
                         dist[solution[-1], solution[0]] for solution in solutions])
        if costs.min() < best_cost:
            best_cost = costs.min()
            best_solution = solutions[np.argmin(costs)]
        pheromone_delta = np.zeros((n, n))
        for solution in solutions:
            for i in range(n-1):
                pheromone_delta[solution[i], solution[i+1]
                                ] += 1.0 / costs[solutions.index(solution)]
            pheromone_delta[solution[-1], solution[0]
                            ] += 1.0 / costs[solutions.index(solution)]
        tau = (1 - rho) * tau + rho * pheromone_delta

    return best_solution, best_cost


def main():
    dist = np.array([[0, 69, 12, 56, 57, 98, 21, 18, 37, 12, 11, 84, 76, 69, 45, 17, 73, 94, 26, 25],
                     [69, 0, 60, 44, 24, 80, 43, 88, 75, 79, 12,
                         65, 28, 11, 47, 16, 78, 41, 42, 23],
                     [12, 60, 0, 87, 81, 18, 42, 24, 51, 81, 65,
                         26, 29, 10, 93, 76, 70, 96, 18, 85],
                     [56, 44, 87, 0, 87, 97, 57, 99, 54, 36, 90,
                      47, 80, 57, 98, 26, 57, 42, 52, 96],
                     [57, 24, 81, 87, 0, 10, 32, 51, 71, 18, 68,
                      33, 61, 21, 34, 42, 28, 76, 57, 24],
                     [98, 80, 18, 97, 10, 0, 71, 86, 66, 71, 11,
                      78, 42, 78, 27, 57, 13, 27, 48, 75],
                     [21, 43, 42, 57, 32, 71, 0, 60, 15, 46, 11,
                      79, 49, 46, 91, 85, 44, 34, 51, 41],
                     [18, 88, 24, 99, 51, 86, 60, 0, 37, 60, 63,
                      95, 48, 41, 75, 68, 73, 63, 35, 96],
                     [37, 75, 51, 54, 71, 66, 15, 37, 0, 53, 96,
                      20, 91, 18, 63, 95, 58, 67, 59, 72],
                     [12, 79, 81, 36, 18, 71, 46, 60, 53, 0, 10,
                      25, 10, 28, 44, 31, 26, 76, 34, 95],
                     [11, 12, 65, 90, 68, 11, 11, 63, 96, 10,
                      0, 82, 20, 57, 89, 39, 53, 55, 33, 60],
                     [84, 65, 26, 47, 33, 78, 79, 95, 20, 25,
                      82, 0, 66, 89, 70, 47, 39, 33, 22, 34],
                     [76, 28, 29, 80, 61, 42, 49, 48, 91, 10,
                      20, 66, 0, 12, 50, 60, 55, 41, 69, 67],
                     [69, 11, 10, 57, 21, 78, 46, 41, 18, 28,
                      57, 89, 12, 0, 42, 63, 15, 56, 16, 72],
                     [45, 47, 93, 98, 34, 27, 91, 75, 63, 44,
                      89, 70, 50, 42, 0, 17, 46, 95, 66, 71],
                     [17, 16, 76, 26, 42, 57, 85, 68, 95, 31,
                      39, 47, 60, 63, 17, 0, 33, 32, 45, 31],
                     [73, 78, 70, 57, 28, 13, 44, 73, 58, 26,
                      53, 39, 55, 15, 46, 33, 0, 75, 54, 67],
                     [94, 41, 96, 42, 76, 27, 34, 63, 67, 76,
                      55, 33, 41, 56, 95, 32, 75, 0, 29, 67],
                     [26, 42, 18, 52, 57, 48, 51, 35, 59, 34,
                      33, 22, 69, 16, 66, 45, 54, 29, 0, 95],
                     [25, 23, 85, 96, 24, 75, 41, 96, 72, 95, 60, 34, 67, 72, 71, 31, 67, 67, 95, 0]])

    print("\nMatriz de distancias: ")
    print(dist)

    # Chame a função ant_system_tsp para encontrar a melhor solução
    start_time = time.time()
    best_solution, best_cost = ant_system_tsp(
        dist, ants=10, max_iter=100, rho=0.5, alpha=1, beta=5, q0=0.9)
    end_time = time.time()

    tempo_de_execução = end_time - start_time

    # Imprima a melhor solução e seu custo
    print("Melhor solução encontrada:", best_solution)
    print("Custo da melhor solução:", best_cost)
    print("Tempo de execução:", tempo_de_execução)


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import time


class SA(object):
    def __init__(self, n, max_iteracoes, max_temperatura, T):
        self.n = n                                   # número de cidades
        self.max_iteracoes = max_iteracoes             # número máximo de iterações
        self.max_temperatura = max_temperatura         # número máximo de temperaturas
        # coordenadas x das cidades
        self.x = np.random.randint(20, 100, self.n)
        # coordenadas y das cidades
        self.y = np.random.randint(20, 100, self.n)
        self.D = np.zeros((self.n, self.n))            # matriz de distâncias
        self.caminho = np.random.permutation(
            self.n)   # cria um caminho aleatório
        # vetor de custo para cada iteração
        self.melhor_custo = np.zeros((self.max_iteracoes, 1))
        self.T = T                                   # temperatura inicial

        # calcula a matriz de distâncias
        for i in range(0, self.n-1):
            for j in range(self.n):
                self.D[i, j] = np.sqrt(
                    (self.x[i]-self.x[j])**2+(self.y[i]-self.y[j])**2)
                self.D[j, i] = self.D[i, j]


def troca(route):
    # gera duas posições aleatórias no caminho e troca as cidades correspondentes
    ix = np.random.permutation(len(route))
    i1 = ix[1]
    i2 = ix[2]
    newroute = np.copy(route)
    newroute[i1], newroute[i2] = newroute[i2], newroute[i1]
    return newroute


def custo(tsp, route):
    L = 0
    route = np.append(route, route[0])
    # calcula o custo do caminho passado como parâmetro
    for i in range(tsp.n):
        L = L+tsp.D[route[i], route[i+1]]
    return L


def solucao(tsp): 
    sol_caminho = tsp.caminho
    sol_custo = custo(tsp, sol_caminho)

    bestsol_caminho = sol_caminho
    bestsol_custo = sol_custo

    for it in range(tsp.max_iteracoes):
        for it2 in range(tsp.max_temperatura):
            newsol_caminho = troca(sol_caminho)
            newsol_custo = custo(tsp, newsol_caminho)

            if newsol_custo < sol_custo:
                sol_custo = newsol_custo
                sol_caminho = newsol_caminho

            else:
                delta = (newsol_custo - sol_custo)
                p = np.exp(-delta/tsp.T)

                if np.random.rand() < p:
                    sol_custo = newsol_custo
                    sol_caminho = newsol_caminho

            if sol_custo < bestsol_custo:
                bestsol_custo = sol_custo
                bestsol_caminho = sol_caminho

        tsp.melhor_custo[it] = bestsol_custo
        tsp.T = tsp.T*0.99

    print("Melhor rota:", bestsol_caminho)
    print("Total Distance: {} km".format(bestsol_custo.round(3)))


def main():
    TSP = SA(n=30, max_iteracoes=2500, max_temperatura=30, T=10000)
    start_time = time.time()
    solucao(TSP)
    end_time = time.time()
    tempo_de_execução = end_time - start_time
    print("Tempo de execução:", tempo_de_execução)


if __name__ == "__main__":
    main()

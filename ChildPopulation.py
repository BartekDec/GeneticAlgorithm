import math
import numpy as np


class ChildPopulation:
    """Selection, cross, mutation """
    @staticmethod
    def roulette(pop, evaluated_pop):
        """ Selection using roulette method
        pop - population of coded units
        evaluated_pop - 'ndarray' object with units evaluation
        new_pop - new population of selected units
        """
        new_pop = []
        total = np.sum(evaluated_pop)
        list_pom = []
        n = 0
        for i in evaluated_pop:
            list_pom.append(i / total)
        list_pom = np.cumsum(list_pom)
        for i in range(len(list_pom)):
            pom = np.random.random()
            for j in range(len(list_pom)):
                if list_pom[j] > pom:
                    new_pop.append(pop[j])
                    break
        new_pop = np.array(new_pop)

        return new_pop

    @staticmethod
    def cross(pop, pk):
        """ Single point cross
        pop - population of coded units
        pk - cross probability for couple units, range [0,1]
        new_pop - new population after cross
        """
        new_pop = np.ndarray(pop.shape, dtype=np.ndarray)
        length = len(pop[0])

        for i in range(len(pop)):
            pom = np.random.random()
            if pom < pk:
                pom2 = np.random.randint(1, high=length)
                if i + 1 < len(pop):
                    n1 = pop[i][0:pom2]
                    n2 = pop[i + 1][pom2:length]
                    n3 = np.concatenate((n1, n2))
                    new_pop[i] = n3

                else:
                    n1 = pop[i][0:pom2]
                    n2 = pop[0][pom2:length]
                    n3 = np.concatenate((n1, n2))
                    new_pop[i] = n3
            else:
                new_pop[i] = pop[i]

        return new_pop

    @staticmethod
    def mutate(pop, pm):
        """ Binary mutation
        pop - population of coded units
        pm - mutation probability for single bit, range [0,1]
        new_pop - new population after mutation
        """
        new_pop = np.ndarray(pop.shape, dtype=np.ndarray)
        for i in range(len(pop)):
            new_pop[i] = pop[i]
            for n in range(len(pop[i])):
                rand = np.random.random()
                if rand < pm:
                    if pop[i][n] == 1:
                        new_pop[i][n] = 0
                    else:
                        new_pop[i][n] = 1
        return new_pop

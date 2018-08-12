import math
import numpy as np


class InitialPopulation:
    @staticmethod
    def nbits(a, b, dx):
        """Number of bits needed to code real number from range [a, b] with 'dx' step
        Method return number of bits and new step
             a - beginning of range
             b - ending of range
             dx - step, accuracy
             B - number of bits
             dx_new - new step
             """
        dif = b - a
        dif = math.fabs(dif)
        n = int(math.ceil(dif / dx))
        B = n.bit_length()
        dx_new = dif / (2 ** B - 1)
        return B, dx_new

    @staticmethod
    def gen_population(P, N, B):
        """ Generate population of random binary units. Population stored in
        'ndarray' object
        P - number of units
        N - number of variables
        B - number of bits for each variable
        pop - population of coded units
        """
        pop = np.ndarray(shape=(P, N * B), dtype=int)
        for i in range(P):
            for j in range(N * B):
                pop[i][j] = np.random.randint(2)
        return pop

    @staticmethod
    def decode_individual(individual, N, B, a, dx):
        """Method to decode units, converting from binary to decimal
        individual - binary unit coding 'N' variables, 'ndarray' object
        N - number of variables
        B - number of bits for each variable
        a - beginning of range
        dx - step, accuracy
        decode_individual - decode unit, 'ndarray' object with 'N' variables
        """
        indiv = np.array(individual, dtype=str).reshape(N, B)
        indiv_new = []
        for i in range(N):
            total = 0
            for j in range(B):
                total += int(indiv[i][j]) * (2 ** (B - 1 - j))
            indiv_new.append(total)
        decode_individual = np.array(indiv_new).__mul__(dx).__add__(a)
        return decode_individual

    @staticmethod
    def evaluate_population(func, pop, N, B, a, dx):
        """ Evaluate units in population, running 'func' on each unit
        func - goal function
        pop - population of coded units
        N - number of variables
        B - number of bits for each variable
        a - beginning of range
        dx - step, accuracy
        evaluated_pop - 'ndarray' object stored 'func' values for each units
        """
        population = []
        for i in range(len(pop)):
            population.append(InitialPopulation.decode_individual(pop[i], N, B, a, dx))
        for j in range(len(population)):
            population[j] = func(population[j])
        evaluated_pop = np.array(population)
        return evaluated_pop

    @staticmethod
    def get_best(pop, evaluated_pop):
        """ Method return best unit from population(max unit)
        pop - population of coded units
        evaluated_pop - 'ndarray' object with units evaluation
        best_individual - best coded unit, array
        best_value - value of best unit
        """
        best_value = max(evaluated_pop)
        best_individual = pop[np.argmax(evaluated_pop)]
        return best_individual, best_value

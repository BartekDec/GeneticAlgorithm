import math
import numpy as np
from InitialPopulation import InitialPopulation
from ChildPopulation import ChildPopulation


class AlgorithmMainLoop:
    def __init__(self, fun, pop_size, pk, pm, generations, dx):
        """ Starting values:
        fun - function which maximum need to be found
        pop_size - population size
        pk - cross probability
        pm - mutation probability
        generations - number of population
        dx - accuracy

        Return values:
        best_sol - best find solution
        best_generation - generation number from which comes best solution
        list_best - list with best unit evaluation from each generation
        list_best_generation - list with best evaluation from each generation, best in population
        list_mean - list with average values evaluations from each generation
        """
        self.initPopulation = InitialPopulation
        self.childPopulation = ChildPopulation
        self.fun = fun
        self.pop_size = pop_size
        self.pk = pk
        self.pm = pm
        self.generations = generations
        self.dx = dx
        self.begin = -500.0
        self.end = 500.0
        self.N = 2
        self.B = None
        self.pop = None
        self.dx_new = None
        self.best_sol = None
        self.best_generation = None
        self.list_best = None
        self.list_best_generation = None
        self.list_mean = None

    @staticmethod
    def sample_function(x):
        """ Sample Schwefel Function """
        d = len(x)
        sum = 0
        for i in x:
            sum = sum + i * np.sin((np.sqrt(np.fabs(i))))
        y = 418.9829 * d - sum
        return y

    def evolution(self):
        """1.Number of bits needed"""
        self.B, self.dx_new = InitialPopulation.nbits(self.begin, self.end, self.dx)

        """2.Generate population"""
        self.pop = InitialPopulation.gen_population(self.pop_size, self.N, self.B)

        """3.First evaluation"""
        evaluated_pop = InitialPopulation.evaluate_population(self.fun, self.pop, self.N, self.B, self.begin,
                                                              self.dx_new)
        """4.First statistics"""
        temp, self.best_sol = InitialPopulation.get_best(self.pop, evaluated_pop)
        self.list_mean = np.array(np.mean(evaluated_pop, 0))
        self.list_best = np.array(self.best_sol)
        self.list_best_generation = np.array(self.best_sol)

        """5.Main loop"""
        for gen in self.generations:
            """6.Selection, cross, mutation"""
            self.pop = ChildPopulation.roulette(self.pop, evaluated_pop)
            self.pop = ChildPopulation.cross(self.pop, self.pk)
            self.pop = ChildPopulation.mutate(self.pop, self.pm)

            """7.Units evaluation"""
            evaluated_pop = InitialPopulation.evaluate_population(self.fun, self.pop)

            """8.Choose best"""
            temp, new_best = InitialPopulation.get_best(self.pop, evaluated_pop)

            """9.Add to list"""
            self.list_best_generation = np.append(self.list_best_generation, new_best)

            """10.Check best units"""
            if new_best < self.best_sol:
                self.best_sol = new_best
                self.best_generation = gen

            """11.List actualization"""
            self.list_best = np.append(self.list_best, self.best_sol)
            self.list_mean = np.append(self.list_mean, np.mean(evaluated_pop, 0))
        return self.best_sol, self.best_generation, self.list_best, self.list_best_generation, self.list_mean

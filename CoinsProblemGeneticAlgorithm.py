import numpy as np
import copy
from colorama import Fore

class Individual:
    def __init__(self, alleles, gen_length, cromosome):
        self._alleles = alleles
        self._gen_length = gen_length
        self._cromosome = cromosome
        self._fitness_score = 0

class GeneticAlgorithm:
    def __init__(self, population, alleles, gen_size, generations, mutation_rate, problem):
        self._population = population
        self._alleles = alleles
        self._gen_size = gen_size
        self._generations = generations
        self._mutation_rate = mutation_rate
        self._problem = problem
        self._individuals = np.array([])

    def run(self):
        self.create_individuals()
        self._best_individual = self._individuals[0]
        generation = 1

        while generation <= self._generations:
            self.fitness_evaluation()
            children = np.array([])

            while len(children) < len(self._individuals):
                parent_1 = self.roulette()
                parent_2 = self.roulette()

                while parent_1 == parent_2:
                    parent_2 = self.roulette()

                child_1, child_2 = self.crossover(self._individuals[parent_1], self._individuals[parent_2])
                children = np.append(children, [child_1])
                children = np.append(children, [child_2])

            self.mutate(children)
            self._individuals = np.copy(children)
            self._individuals[np.random.randint(len(self._individuals))] = copy.deepcopy(self._best_individual)

            if generation % 100 == 0:
                print("Generación: ", generation, 'Mejor Histórico: ', self._best_individual._cromosome, self._best_individual._fitness_score)

            generation += 1

    def create_individuals(self):
        for individual in range(self._population):
            cromosome = np.random.randint(2, size = self._alleles)
            individual = Individual(self._alleles, self._gen_size, cromosome)
            self._individuals = np.append(self._individuals, [individual])

    def fitness_evaluation(self):
        for individual in self._individuals:
            individual._fitness_score = self._problem.get_fitness_score(self._gen_size, individual._cromosome)
            if individual._fitness_score > self._best_individual._fitness_score:
                self._best_individual = copy.deepcopy(individual)

    def roulette(self):
        initial_fitness_score_sum = np.sum([individual._fitness_score for individual in self._individuals])

        if initial_fitness_score_sum < 0:
            initial_fitness_score_sum *= -1

        if initial_fitness_score_sum == 0:
            return np.random.randint(len(self._individuals))
        else:
            threshold = np.random.randint(initial_fitness_score_sum + 1)
            selected_individual = 0
            current_fitness_score_sum = self._individuals[selected_individual]._fitness_score

            if current_fitness_score_sum < 0:
                current_fitness_score_sum *= -1

            while current_fitness_score_sum < threshold and selected_individual < (len(self._individuals) - 1):
                selected_individual += 1

                if self._individuals[selected_individual]._fitness_score < 0:
                    current_fitness_score_sum += self._individuals[selected_individual]._fitness_score * -1
                else:
                    current_fitness_score_sum += self._individuals[selected_individual]._fitness_score

            return selected_individual

    def crossover(self, individual_1, individual_2):
        child_1 = copy.deepcopy(individual_1)
        child_2 = copy.deepcopy(individual_2)

        max_crossover_point = self._alleles - 1
        crossover_point = np.random.randint(max_crossover_point) + 1
        child_1._cromosome[crossover_point:], child_2._cromosome[crossover_point:] = child_2._cromosome[crossover_point:], child_1._cromosome[crossover_point:]

        return child_1, child_2

    def mutate(self, children):
        for child in children:
            for allele in range(len(child._cromosome)):
                if np.random.rand() < self._mutation_rate:
                    child._cromosome[allele] = int(not child._cromosome[allele])

    def get_best_solution(self):
        return self._best_individual

class CoinsProblem:
    def __init__(self, coins):
        self._coins = coins
        self._allele_weight = 1 / len(coins)
        self._fitness_score = 0

    def check_next_allele(self, allele_number):
      return self._coins[allele_number] != self._coins[allele_number + 1]

    def check_previous_allele(self, allele_number):
      return self._coins[allele_number - 1] != self._coins[allele_number]

    def check_previous_and_next_alleles(self, allele_number):
      return self.check_previous_allele(allele_number) and self.check_next_allele(allele_number)

    def get_reward(self):
      return self._fitness_score + self._allele_weight

    def get_penalty(self):
      return self._fitness_score - self._allele_weight

    def get_fitness_score(self, gen_size, cromosome):
        self._fitness_score = 0

        cromosome_length = len(cromosome)
        for allele_number in range(cromosome_length):

          if cromosome[allele_number] == 0:
            continue

          if allele_number == 0:
            if (cromosome_length - 1) >= (allele_number + 1):
              self._fitness_score = self.get_reward() if self.check_next_allele(allele_number) else self.get_penalty()
            else:
              self._fitness_score = self.get_reward()
          else:
            if allele_number == cromosome_length - 1:
              self._fitness_score = self.get_reward() if self.check_previous_allele(allele_number) else self.get_penalty()
            else:
              self._fitness_score = self.get_reward() if self.check_previous_and_next_alleles(allele_number) else self.get_penalty()

        return self._fitness_score

def main():
    coins = [ 1, 20, 5, 1, 2, 5, 5, 1, 5, 2, 2, 1, 10, 5, 10, 5, 20, 20, 20, 5, 1, 1, 20, 20, 1, 10, 2, 10, 5, 2, 10, 1, 20, 1, 20, 10, 5, 5, 20, 2, 10, 1, 2, 5, 10, 20, 10, 2, 5, 5, 20, 1, 1, 5, 10, 10, 10, 1, 5, 2, 1, 2, 10, 20, 2, 10, 10, 20, 5, 10, 1, 2, 1, 5, 20, 2, 5, 1, 5, 10, 2, 5, 10, 2, 1, 1, 1, 10, 20, 10, 20, 2, 2, 10, 20, 10, 1, 1, 5, 2]
    coins_problem = CoinsProblem(coins)
    alleles = len(coins)
    individuals = 20
    gen_size = 1
    generations = 1400
    mutation_rate = 0.01
    genetic_algorithm = GeneticAlgorithm(individuals, alleles, gen_size, generations, mutation_rate, coins_problem)
    genetic_algorithm.run()
    best_solution = genetic_algorithm.get_best_solution()

    for coin in range(len(coins)):
      if best_solution._cromosome[coin]:
        print(f"{Fore.GREEN}{coins[coin]}{Fore.RED} ", end="")
      else:
        print(f"{Fore.RED}{coins[coin]}{Fore.RED} ", end="")

main()

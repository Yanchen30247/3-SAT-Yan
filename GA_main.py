"""
Genetic Algorithm Main Flow for 3-SAT Problem

This module implements a standard GA baseline with:
- Hybrid initialization (random + heuristic)
- Tournament/Roulette selection with elitism
- Multiple crossover operators
- Directed mutation
- Local search (GSAT/WalkSAT style)
- Diversity restart mechanism

This serves as the baseline for comparison with GA+WoC variants.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter

# Import from our modules
from solve_and_eval import SATInstance, SATSolution, SATEvaluator, read_dimacs_cnf


class GeneticAlgorithm:
    """Standard Genetic Algorithm for 3-SAT."""

    def __init__(self,
                 instance: SATInstance,
                 population_size: int = 100,
                 elite_size: int = 2,
                 tournament_size: int = 3,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.01,
                 max_generations: int = 1000,
                 stall_generations: int = 50,
                 local_search_steps: int = 20,
                 local_search_elite: int = 5,
                 use_weighted_fitness: bool = True,
                 verbose: bool = True):
        """
        Initialize GA.

        Args:
            instance: SAT instance to solve
            population_size: Size of population (50-200 recommended)
            elite_size: Number of elite individuals to preserve
            tournament_size: Tournament size for selection (2-4)
            crossover_rate: Probability of crossover
            mutation_rate: Base mutation rate (0.005-0.02)
            max_generations: Maximum number of generations
            stall_generations: Generations without improvement before restart
            local_search_steps: Steps of local search per elite
            local_search_elite: Number of elite to apply local search
            use_weighted_fitness: Use weighted clause fitness
            verbose: Print progress information
        """
        self.instance = instance
        self.evaluator = SATEvaluator(instance, use_weighted=use_weighted_fitness)

        # GA parameters
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.stall_generations = stall_generations
        self.local_search_steps = local_search_steps
        self.local_search_elite = local_search_elite
        self.verbose = verbose

        # Population and statistics
        self.population: List[SATSolution] = []
        self.fitness_cache: Dict[str, Tuple[float, int, bool]] = {}
        self.best_solution: SATSolution = None
        self.best_fitness: float = -1
        self.best_generation: int = 0

        # Diversity tracking
        self.pattern_history: List[Counter] = []

        # Statistics
        self.generation = 0
        self.total_evaluations = 0
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    def initialize_population(self):
        """
        Initialize population with 50% random + 50% heuristic solutions.
        """
        self.population = []
        half = self.population_size // 2

        # 50% pure random
        for _ in range(half):
            solution = SATSolution(self.instance.num_vars)
            self.population.append(solution)

        # 50% heuristic-based
        for _ in range(self.population_size - half):
            solution = self._create_heuristic_solution()
            self.population.append(solution)

        if self.verbose:
            print(f"Initialized population of {self.population_size} individuals")
            print(f"  - {half} random solutions")
            print(f"  - {self.population_size - half} heuristic solutions")

    def _create_heuristic_solution(self) -> SATSolution:
        """
        Create solution using heuristics:
        - Process unit clauses
        - Handle pure literals
        - Random for remaining
        """
        assignment = [None] * self.instance.num_vars

        # Find unit clauses (clauses with single literal)
        for clause in self.instance.clauses:
            if len(clause) == 1:
                literal = clause[0]
                var_idx = abs(literal) - 1
                assignment[var_idx] = literal > 0

        # Find pure literals (appear only positive or only negative)
        literal_polarity = {}
        for clause in self.instance.clauses:
            for literal in clause:
                var = abs(literal)
                if var not in literal_polarity:
                    literal_polarity[var] = set()
                literal_polarity[var].add(literal > 0)

        # Assign pure literals
        for var, polarities in literal_polarity.items():
            if len(polarities) == 1:
                var_idx = var - 1
                if assignment[var_idx] is None:
                    assignment[var_idx] = list(polarities)[0]

        # Frequency-based heuristic for remaining
        var_frequency = Counter()
        for clause in self.instance.clauses:
            for literal in clause:
                var_frequency[literal] += 1

        for i in range(len(assignment)):
            if assignment[i] is None:
                # Bias towards more frequent polarity
                pos_freq = var_frequency.get(i + 1, 0)
                neg_freq = var_frequency.get(-(i + 1), 0)

                if pos_freq + neg_freq == 0:
                    assignment[i] = random.choice([True, False])
                else:
                    prob_true = pos_freq / (pos_freq + neg_freq)
                    assignment[i] = random.random() < prob_true

        return SATSolution(self.instance.num_vars, assignment)

    def evaluate_population(self):
        """Evaluate all individuals and update best solution."""
        for solution in self.population:
            key = solution.to_string()
            if key not in self.fitness_cache:
                fitness, flips, is_sat = self.evaluator.evaluate(solution)
                self.fitness_cache[key] = (fitness, flips, is_sat)
                self.total_evaluations += 1

                # Update best solution
                if fitness > self.best_fitness or \
                   (fitness == self.best_fitness and flips < self.fitness_cache.get(self.best_solution.to_string(), (0, float('inf'), False))[1]):
                    self.best_solution = solution.copy()
                    self.best_fitness = fitness
                    self.best_generation = self.generation

                    # Check if solved
                    if is_sat:
                        if self.verbose:
                            print(f"\n*** SOLUTION FOUND at generation {self.generation} ***")
                        return True

        return False

    def tournament_selection(self) -> SATSolution:
        """
        Tournament selection.

        Returns:
            Selected solution
        """
        tournament = random.sample(self.population, self.tournament_size)
        best = tournament[0]

        # Get fitness, evaluate if not cached
        best_key = best.to_string()
        if best_key not in self.fitness_cache:
            best_fit, best_flips, _ = self.evaluator.evaluate(best)
            self.fitness_cache[best_key] = (best_fit, best_flips, _)
            self.total_evaluations += 1
        else:
            best_fit, best_flips, _ = self.fitness_cache[best_key]

        for solution in tournament[1:]:
            sol_key = solution.to_string()
            if sol_key not in self.fitness_cache:
                fit, flips, is_sat = self.evaluator.evaluate(solution)
                self.fitness_cache[sol_key] = (fit, flips, is_sat)
                self.total_evaluations += 1
            else:
                fit, flips, _ = self.fitness_cache[sol_key]

            if fit > best_fit or (fit == best_fit and flips < best_flips):
                best = solution
                best_fit = fit
                best_flips = flips

        return best.copy()

    def roulette_selection(self) -> SATSolution:
        """
        Roulette wheel selection.

        Returns:
            Selected solution
        """
        # Get all fitness values
        fitness_values = []
        for solution in self.population:
            sol_key = solution.to_string()
            if sol_key not in self.fitness_cache:
                fit, flips, is_sat = self.evaluator.evaluate(solution)
                self.fitness_cache[sol_key] = (fit, flips, is_sat)
                self.total_evaluations += 1
            else:
                fit, _, _ = self.fitness_cache[sol_key]
            fitness_values.append(max(0, fit))  # Ensure non-negative

        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.choice(self.population).copy()

        # Roulette wheel
        pick = random.uniform(0, total_fitness)
        current = 0
        for solution, fitness in zip(self.population, fitness_values):
            current += fitness
            if current >= pick:
                return solution.copy()

        return self.population[-1].copy()

    def one_point_crossover(self, parent1: SATSolution, parent2: SATSolution) -> Tuple[SATSolution, SATSolution]:
        """One-point crossover."""
        point = random.randint(1, self.instance.num_vars - 1)

        child1_assign = parent1.assignment[:point] + parent2.assignment[point:]
        child2_assign = parent2.assignment[:point] + parent1.assignment[point:]

        return (SATSolution(self.instance.num_vars, child1_assign),
                SATSolution(self.instance.num_vars, child2_assign))

    def two_point_crossover(self, parent1: SATSolution, parent2: SATSolution) -> Tuple[SATSolution, SATSolution]:
        """Two-point crossover."""
        point1 = random.randint(1, self.instance.num_vars - 2)
        point2 = random.randint(point1 + 1, self.instance.num_vars - 1)

        child1_assign = (parent1.assignment[:point1] +
                        parent2.assignment[point1:point2] +
                        parent1.assignment[point2:])
        child2_assign = (parent2.assignment[:point1] +
                        parent1.assignment[point1:point2] +
                        parent2.assignment[point2:])

        return (SATSolution(self.instance.num_vars, child1_assign),
                SATSolution(self.instance.num_vars, child2_assign))

    def uniform_crossover(self, parent1: SATSolution, parent2: SATSolution,
                         bias_elite: bool = True) -> Tuple[SATSolution, SATSolution]:
        """
        Uniform crossover with optional elite bias.

        Args:
            parent1: First parent
            parent2: Second parent
            bias_elite: Whether to bias towards elite variable assignments
        """
        child1_assign = []
        child2_assign = []

        # Compute elite patterns if needed
        elite_pattern = None
        if bias_elite and len(self.pattern_history) > 0:
            elite_pattern = self.pattern_history[-1]

        for i in range(self.instance.num_vars):
            # Determine bias
            prob = 0.5
            if elite_pattern is not None:
                true_count = elite_pattern.get((i, True), 0)
                false_count = elite_pattern.get((i, False), 0)
                total = true_count + false_count
                if total > 0:
                    prob = true_count / total

            # Crossover with bias
            if random.random() < prob:
                child1_assign.append(parent1.assignment[i])
                child2_assign.append(parent2.assignment[i])
            else:
                child1_assign.append(parent2.assignment[i])
                child2_assign.append(parent1.assignment[i])

        return (SATSolution(self.instance.num_vars, child1_assign),
                SATSolution(self.instance.num_vars, child2_assign))

    def crossover(self, parent1: SATSolution, parent2: SATSolution) -> Tuple[SATSolution, SATSolution]:
        """
        Perform crossover with random operator selection.
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        operator = random.choice(['one_point', 'two_point', 'uniform'])

        if operator == 'one_point':
            return self.one_point_crossover(parent1, parent2)
        elif operator == 'two_point':
            return self.two_point_crossover(parent1, parent2)
        else:
            return self.uniform_crossover(parent1, parent2)

    def directed_mutation(self, solution: SATSolution):
        """
        Directed mutation: prefer flipping variables in unsatisfied clauses.
        """
        _, unsatisfied = self.evaluator.count_satisfied_clauses(solution)

        if len(unsatisfied) > 0:
            # Collect variables in unsatisfied clauses
            unsat_vars = set()
            for idx in unsatisfied:
                clause = self.instance.clauses[idx]
                for literal in clause:
                    unsat_vars.add(abs(literal) - 1)

            # Higher probability for variables in unsatisfied clauses
            for var_idx in range(self.instance.num_vars):
                if var_idx in unsat_vars:
                    # Higher mutation rate for problematic variables
                    if random.random() < self.mutation_rate * 3:
                        solution.flip(var_idx)
                else:
                    # Normal mutation rate
                    if random.random() < self.mutation_rate:
                        solution.flip(var_idx)
        else:
            # Standard bit-flip mutation
            for var_idx in range(self.instance.num_vars):
                if random.random() < self.mutation_rate:
                    solution.flip(var_idx)

    def local_search_gsat(self, solution: SATSolution, max_steps: int):
        """
        GSAT-style local search: flip variable that maximally improves fitness.
        """
        for _ in range(max_steps):
            best_var, best_gain = self.evaluator.get_best_flip(solution)

            if best_gain > 0:
                solution.flip(best_var)
            else:
                # Random walk step
                if random.random() < 0.1:
                    solution.flip(random.randint(0, self.instance.num_vars - 1))
                else:
                    break

    def apply_local_search(self):
        """Apply local search to top elite solutions."""
        # Sort population by fitness
        sorted_pop = sorted(self.population,
                          key=lambda s: self.fitness_cache[s.to_string()],
                          reverse=True)

        # Apply local search to top elite
        for i in range(min(self.local_search_elite, len(sorted_pop))):
            # Store the old key before modification
            old_key = sorted_pop[i].to_string()

            # Apply local search (this modifies the solution)
            self.local_search_gsat(sorted_pop[i], self.local_search_steps)

            # Remove old cached value and re-evaluate
            if old_key in self.fitness_cache:
                del self.fitness_cache[old_key]

            # Re-evaluate the modified solution
            new_key = sorted_pop[i].to_string()
            if new_key not in self.fitness_cache:
                fitness, flips, is_sat = self.evaluator.evaluate(sorted_pop[i])
                self.fitness_cache[new_key] = (fitness, flips, is_sat)
                self.total_evaluations += 1

    def compute_diversity(self) -> float:
        """
        Compute population diversity (average hamming distance).
        """
        if len(self.population) < 2:
            return 0.0

        total_distance = 0
        count = 0

        for i in range(len(self.population)):
            for j in range(i + 1, min(i + 20, len(self.population))):  # Sample for efficiency
                hamming = sum(a != b for a, b in zip(
                    self.population[i].assignment,
                    self.population[j].assignment
                ))
                total_distance += hamming
                count += 1

        return total_distance / (count * self.instance.num_vars) if count > 0 else 0.0

    def update_pattern_history(self):
        """Track elite patterns for diversity analysis."""
        # Sort population
        sorted_pop = sorted(self.population,
                          key=lambda s: self.fitness_cache[s.to_string()],
                          reverse=True)

        # Count patterns in elite
        pattern_counter = Counter()
        for i in range(min(self.elite_size * 2, len(sorted_pop))):
            solution = sorted_pop[i]
            for var_idx, value in enumerate(solution.assignment):
                pattern_counter[(var_idx, value)] += 1

        self.pattern_history.append(pattern_counter)

        # Keep only recent history
        if len(self.pattern_history) > 20:
            self.pattern_history.pop(0)

    def diversity_restart(self):
        """
        Diversity restart: keep elite + regenerate rest from uncommon patterns.
        """
        if self.verbose:
            print(f"\n>>> Diversity restart at generation {self.generation}")

        # Sort and keep elite
        sorted_pop = sorted(self.population,
                          key=lambda s: self.fitness_cache[s.to_string()],
                          reverse=True)

        elite = sorted_pop[:self.elite_size]

        # Regenerate rest
        new_pop = elite.copy()

        # Use pattern history to generate diverse solutions
        while len(new_pop) < self.population_size:
            if len(self.pattern_history) > 0 and random.random() < 0.5:
                # Generate from uncommon patterns
                solution = self._create_diverse_solution()
            else:
                # Random or heuristic
                if random.random() < 0.5:
                    solution = SATSolution(self.instance.num_vars)
                else:
                    solution = self._create_heuristic_solution()

            new_pop.append(solution)

        self.population = new_pop

        # Clear fitness cache for new individuals
        for solution in self.population[self.elite_size:]:
            key = solution.to_string()
            if key in self.fitness_cache:
                del self.fitness_cache[key]

    def _create_diverse_solution(self) -> SATSolution:
        """Create solution from uncommon patterns in history."""
        assignment = []

        if len(self.pattern_history) > 0:
            recent_patterns = self.pattern_history[-1]

            for var_idx in range(self.instance.num_vars):
                true_count = recent_patterns.get((var_idx, True), 0)
                false_count = recent_patterns.get((var_idx, False), 0)

                # Bias towards less common value
                if true_count + false_count > 0:
                    prob_true = false_count / (true_count + false_count)
                else:
                    prob_true = 0.5

                assignment.append(random.random() < prob_true)
        else:
            assignment = [random.choice([True, False]) for _ in range(self.instance.num_vars)]

        return SATSolution(self.instance.num_vars, assignment)

    def evolve_generation(self) -> bool:
        """
        Evolve one generation.

        Returns:
            True if solution found, False otherwise
        """
        # Evaluate population
        if self.evaluate_population():
            return True

        # Sort population by fitness
        sorted_pop = sorted(self.population,
                          key=lambda s: self.fitness_cache[s.to_string()],
                          reverse=True)

        # Statistics
        avg_fitness = np.mean([self.fitness_cache[s.to_string()][0] for s in self.population])
        diversity = self.compute_diversity()
        self.fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)

        # Update pattern history
        self.update_pattern_history()

        if self.verbose and self.generation % 10 == 0:
            print(f"Gen {self.generation}: Best={self.best_fitness:.4f}, "
                  f"Avg={avg_fitness:.4f}, Diversity={diversity:.4f}, "
                  f"Evals={self.total_evaluations}")

        # Apply local search
        self.apply_local_search()

        # Check for stall and restart
        if self.generation - self.best_generation >= self.stall_generations:
            self.diversity_restart()
            self.best_generation = self.generation  # Reset stall counter

        # Create next generation
        next_population = []

        # Elitism: preserve best
        next_population.extend([s.copy() for s in sorted_pop[:self.elite_size]])

        # Generate offspring
        while len(next_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            self.directed_mutation(child1)
            self.directed_mutation(child2)

            next_population.append(child1)
            if len(next_population) < self.population_size:
                next_population.append(child2)

        self.population = next_population
        self.generation += 1

        # Update clause weights for adaptive fitness
        if hasattr(self.evaluator, 'instance'):
            _, unsatisfied = self.evaluator.count_satisfied_clauses(self.best_solution)
            self.instance.update_clause_weights(unsatisfied)

        return False

    def run(self) -> Tuple[SATSolution, float, bool]:
        """
        Run the GA.

        Returns:
            Tuple of (best_solution, best_fitness, is_satisfying)
        """
        if self.verbose:
            print(f"\nStarting Standard GA")
            print(f"Instance: {self.instance.num_vars} vars, {self.instance.num_clauses} clauses")
            print(f"Population: {self.population_size}, Elite: {self.elite_size}")
            print(f"Crossover: {self.crossover_rate}, Mutation: {self.mutation_rate}")
            print("=" * 70)

        # Initialize
        self.initialize_population()

        # Evolution loop
        for gen in range(self.max_generations):
            if self.evolve_generation():
                break

        # Final evaluation
        final_fit, final_flips, is_sat = self.evaluator.evaluate(self.best_solution)

        if self.verbose:
            print("\n" + "=" * 70)
            print(f"GA Completed:")
            print(f"  Generations: {self.generation}")
            print(f"  Total Evaluations: {self.total_evaluations}")
            print(f"  Best Fitness: {final_fit:.4f}")
            print(f"  Satisfying: {is_sat}")
            print(f"  Best found at generation: {self.best_generation}")

        return self.best_solution, final_fit, is_sat


if __name__ == "__main__":
    print("Testing Standard GA for 3-SAT\n")
    print("=" * 70)

    # Test 1: Simple hand-crafted instance
    print("\n[Test 1] Running GA on simple 3-SAT instance...")
    test_clauses = [
        [1, 2, 3],
        [-1, 2, -3],
        [1, -2, 3],
        [-1, -2, -3],
        [1, 2, -3]
    ]
    test_instance = SATInstance(3, test_clauses)

    ga = GeneticAlgorithm(
        test_instance,
        population_size=20,
        elite_size=2,
        max_generations=50,
        verbose=True
    )

    best_sol, best_fit, is_sat = ga.run()
    print(f"\nBest solution: {best_sol.to_string()}")
    print(f"Fitness: {best_fit:.4f}")
    print(f"Satisfying: {is_sat}")

    # Test 2: Load from file if available
    print("\n" + "=" * 70)
    print("\n[Test 2] Testing with generated CNF file...")
    try:
        import os
        test_cnf_path = "data/test/test_3partition_easy.cnf"
        if os.path.exists(test_cnf_path):
            instance_from_file = read_dimacs_cnf(test_cnf_path)
            print(f"Loaded instance: {instance_from_file.num_vars} vars, "
                  f"{instance_from_file.num_clauses} clauses")

            ga_file = GeneticAlgorithm(
                instance_from_file,
                population_size=50,
                elite_size=2,
                max_generations=100,
                stall_generations=30,
                verbose=True
            )

            best_sol_file, best_fit_file, is_sat_file = ga_file.run()
            print(f"\nBest fitness: {best_fit_file:.4f}")
            print(f"Satisfying: {is_sat_file}")

            # Show fitness history
            if len(ga_file.fitness_history) > 0:
                print(f"\nFitness progression (every 10 gens):")
                for i in range(0, len(ga_file.fitness_history), 10):
                    print(f"  Gen {i}: {ga_file.fitness_history[i]:.4f}")
        else:
            print(f"CNF file not found: {test_cnf_path}")
            print("(Run data_generation.py first)")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Testing completed!")

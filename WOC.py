"""
Wisdom of Crowds (WoC) for 3-SAT Problem

This module implements a WoC approach that:
- Builds a diverse expert pool (16-64 GA instances with varied parameters)
- Extracts opinions (top-k elite solutions from each expert)
- Computes variable support with weighted voting
- Performs consensus-based assignment with skeleton extraction
- Handles conflicts using WalkSAT-style repair
- Iterative refinement until convergence
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict
import concurrent.futures
from scipy import stats
import copy

# Import from our modules
from solve_and_eval import SATInstance, SATSolution, SATEvaluator
from GA_main import GeneticAlgorithm


class ExpertConfig:
    """Configuration for a single expert GA."""

    def __init__(self,
                 expert_id: int,
                 population_size: int = 100,
                 elite_size: int = 2,
                 tournament_size: int = 3,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.01,
                 max_generations: int = 100,
                 stall_generations: int = 30,
                 local_search_steps: int = 20,
                 local_search_elite: int = 5,
                 use_weighted_fitness: bool = True,
                 init_strategy: str = "hybrid"):
        """
        Initialize expert configuration.

        Args:
            expert_id: Unique ID for this expert
            population_size: Population size
            elite_size: Number of elite individuals
            tournament_size: Tournament size for selection
            crossover_rate: Crossover probability
            mutation_rate: Mutation rate
            max_generations: Maximum generations
            stall_generations: Generations before restart
            local_search_steps: Steps of local search
            local_search_elite: Number of elite for local search
            use_weighted_fitness: Use weighted fitness
            init_strategy: Initialization strategy ("hybrid", "random", "heuristic")
        """
        self.expert_id = expert_id
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.stall_generations = stall_generations
        self.local_search_steps = local_search_steps
        self.local_search_elite = local_search_elite
        self.use_weighted_fitness = use_weighted_fitness
        self.init_strategy = init_strategy

    @staticmethod
    def create_diverse_configs(num_experts: int) -> List['ExpertConfig']:
        """
        Create diverse expert configurations.

        Args:
            num_experts: Number of experts to create

        Returns:
            List of expert configurations
        """
        configs = []

        for i in range(num_experts):
            # Vary parameters
            pop_size = random.choice([50, 75, 100, 150, 200])
            elite_size = random.choice([1, 2, 3, 5])
            tournament_size = random.choice([2, 3, 4, 5])
            crossover_rate = random.uniform(0.7, 0.95)
            mutation_rate = random.uniform(0.005, 0.03)
            stall_gens = random.choice([20, 30, 40, 50])
            local_steps = random.choice([10, 20, 30, 50])
            local_elite = random.choice([3, 5, 7, 10])
            use_weighted = random.choice([True, False])
            init_strategy = random.choice(["hybrid", "random", "heuristic"])

            config = ExpertConfig(
                expert_id=i,
                population_size=pop_size,
                elite_size=elite_size,
                tournament_size=tournament_size,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                max_generations=100,  # Fixed time budget
                stall_generations=stall_gens,
                local_search_steps=local_steps,
                local_search_elite=local_elite,
                use_weighted_fitness=use_weighted,
                init_strategy=init_strategy
            )
            configs.append(config)

        return configs


class Expert:
    """Individual expert that runs a GA variant."""

    def __init__(self, config: ExpertConfig, instance: SATInstance, seed: int = None):
        """
        Initialize expert.

        Args:
            config: Expert configuration
            instance: SAT instance to solve
            seed: Random seed
        """
        self.config = config
        self.instance = instance
        self.seed = seed

        # Results
        self.elite_solutions: List[SATSolution] = []
        self.elite_fitness: List[float] = []
        self.best_solution: SATSolution = None
        self.best_fitness: float = -1

    def run(self, top_k: int = 5, verbose: bool = False) -> Tuple[List[SATSolution], List[float]]:
        """
        Run the expert GA and extract top-k solutions.

        Args:
            top_k: Number of elite solutions to extract
            verbose: Print progress

        Returns:
            Tuple of (elite_solutions, elite_fitness)
        """
        try:
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)

            # Create and run GA
            ga = GeneticAlgorithm(
                self.instance,
                population_size=self.config.population_size,
                elite_size=self.config.elite_size,
                tournament_size=self.config.tournament_size,
                crossover_rate=self.config.crossover_rate,
                mutation_rate=self.config.mutation_rate,
                max_generations=self.config.max_generations,
                stall_generations=self.config.stall_generations,
                local_search_steps=self.config.local_search_steps,
                local_search_elite=self.config.local_search_elite,
                use_weighted_fitness=self.config.use_weighted_fitness,
                verbose=verbose
            )

            best_sol, best_fit, is_sat = ga.run()
            self.best_solution = best_sol
            self.best_fitness = best_fit

            # Extract top-k elite solutions from final population
            evaluator = SATEvaluator(self.instance, use_weighted=True)

            # Evaluate all population members and cache results
            pop_with_fitness = []
            for sol in ga.population:
                key = sol.to_string()
                # Use GA's cache if available, otherwise evaluate
                if key in ga.fitness_cache:
                    fit = ga.fitness_cache[key][0]  # Get fitness from cache
                else:
                    fit, flips, is_sat_check = evaluator.evaluate(sol)
                pop_with_fitness.append((sol, fit))

            # Sort by fitness and take top-k
            pop_with_fitness.sort(key=lambda x: x[1], reverse=True)

            self.elite_solutions = [sol for sol, fit in pop_with_fitness[:top_k]]
            self.elite_fitness = [fit for sol, fit in pop_with_fitness[:top_k]]

            if verbose:
                print(f"Expert {self.config.expert_id}: Best fitness = {best_fit:.4f}")

            return self.elite_solutions, self.elite_fitness

        except Exception as e:
            if verbose:
                print(f"Expert {self.config.expert_id} failed with error: {e}")
            # Return empty results on failure
            return [], []


class WisdomOfCrowds:
    """Wisdom of Crowds solver for 3-SAT."""

    def __init__(self,
                 instance: SATInstance,
                 num_experts: int = 32,
                 top_k: int = 5,
                 consensus_threshold_high: float = 0.7,
                 consensus_threshold_low: float = 0.3,
                 beta: float = 2.0,
                 max_iterations: int = 3,
                 parallel: bool = True,
                 verbose: bool = True):
        """
        Initialize WoC solver.

        Args:
            instance: SAT instance to solve
            num_experts: Number of expert GAs (16-64)
            top_k: Number of elite solutions per expert
            consensus_threshold_high: Threshold for True assignment
            consensus_threshold_low: Threshold for False assignment
            beta: Weight exponent for fitness-based weighting
            max_iterations: Maximum WoC iterations
            parallel: Use parallel expert execution
            verbose: Print progress
        """
        self.instance = instance
        self.num_experts = num_experts
        self.top_k = top_k
        self.consensus_threshold_high = consensus_threshold_high
        self.consensus_threshold_low = consensus_threshold_low
        self.beta = beta
        self.max_iterations = max_iterations
        self.parallel = parallel
        self.verbose = verbose

        # Expert pool
        self.expert_configs: List[ExpertConfig] = []
        self.experts: List[Expert] = []

        # Results
        self.all_solutions: List[SATSolution] = []
        self.all_fitness: List[float] = []
        self.variable_support: np.ndarray = None
        self.consensus_assignment: List[int] = []  # -1=uncertain, 0=False, 1=True
        self.best_solution: SATSolution = None
        self.best_fitness: float = -1

    def build_expert_pool(self):
        """Build diverse expert pool with varied configurations."""
        self.expert_configs = ExpertConfig.create_diverse_configs(self.num_experts)

        self.experts = []
        for config in self.expert_configs:
            expert = Expert(config, self.instance, seed=config.expert_id)
            self.experts.append(expert)

        if self.verbose:
            print(f"Built expert pool with {self.num_experts} experts")

    def run_experts_parallel(self) -> List[Tuple[List[SATSolution], List[float]]]:
        """Run all experts in parallel."""
        results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.num_experts, 8)) as executor:
            futures = []
            for expert in self.experts:
                future = executor.submit(expert.run, self.top_k, False)
                futures.append(future)

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    elite_sols, elite_fits = future.result()
                    results.append((elite_sols, elite_fits))
                    if self.verbose:
                        print(f"Expert {i} completed")
                except Exception as e:
                    print(f"Expert {i} failed: {e}")
                    results.append(([], []))

        return results

    def run_experts_serial(self) -> List[Tuple[List[SATSolution], List[float]]]:
        """Run all experts serially."""
        results = []

        for i, expert in enumerate(self.experts):
            try:
                elite_sols, elite_fits = expert.run(self.top_k, self.verbose)
                results.append((elite_sols, elite_fits))
            except Exception as e:
                print(f"Expert {i} failed: {e}")
                results.append(([], []))

        return results

    def extract_opinions(self):
        """Extract opinions from all experts."""
        if self.verbose:
            print(f"\nRunning {self.num_experts} experts...")

        # Run experts
        if self.parallel:
            try:
                expert_results = self.run_experts_parallel()
            except:
                if self.verbose:
                    print("Parallel execution failed, falling back to serial")
                expert_results = self.run_experts_serial()
        else:
            expert_results = self.run_experts_serial()

        # Collect all solutions
        self.all_solutions = []
        self.all_fitness = []

        for elite_sols, elite_fits in expert_results:
            self.all_solutions.extend(elite_sols)
            self.all_fitness.extend(elite_fits)

        if self.verbose:
            print(f"Collected {len(self.all_solutions)} elite solutions")

    def compute_solution_weights(self) -> np.ndarray:
        """
        Compute weights for solutions using fitness-based exponential weighting
        and diversity adjustment.

        Returns:
            Array of solution weights
        """
        if len(self.all_fitness) == 0:
            return np.array([])

        # Z-score normalize fitness
        fitness_array = np.array(self.all_fitness)
        if np.std(fitness_array) > 0:
            z_scores = stats.zscore(fitness_array)
        else:
            z_scores = np.zeros_like(fitness_array)

        # Exponential weights
        weights = np.exp(self.beta * z_scores)

        # Diversity-based weight adjustment
        # Reduce weight for similar solutions
        for i in range(len(self.all_solutions)):
            similarity_penalty = 0
            for j in range(len(self.all_solutions)):
                if i != j:
                    # Hamming distance
                    hamming = sum(a != b for a, b in zip(
                        self.all_solutions[i].assignment,
                        self.all_solutions[j].assignment
                    ))
                    similarity = 1.0 - (hamming / self.instance.num_vars)

                    # Penalize if very similar and j has higher fitness
                    if similarity > 0.9 and self.all_fitness[j] > self.all_fitness[i]:
                        similarity_penalty += similarity

            # Apply penalty
            weights[i] *= np.exp(-0.1 * similarity_penalty)

        # Normalize weights
        weights = weights / np.sum(weights)

        return weights

    def compute_variable_support(self):
        """
        Compute support for each variable: p_i = weighted fraction voting True.
        """
        if len(self.all_solutions) == 0:
            return

        weights = self.compute_solution_weights()
        self.variable_support = np.zeros(self.instance.num_vars)

        # Compute weighted support for each variable
        total_weight = np.sum(weights)

        for var_idx in range(self.instance.num_vars):
            support = 0.0
            for sol_idx, solution in enumerate(self.all_solutions):
                if solution.assignment[var_idx]:
                    support += weights[sol_idx]

            self.variable_support[var_idx] = support / total_weight if total_weight > 0 else 0.5

        if self.verbose:
            print(f"Variable support computed: mean={np.mean(self.variable_support):.3f}, "
                  f"std={np.std(self.variable_support):.3f}")

    def extract_consensus_assignment(self) -> Tuple[List[int], List[int]]:
        """
        Extract consensus assignment using thresholds.

        Returns:
            Tuple of (consensus_assignment, uncertain_vars)
            - consensus_assignment: -1=uncertain, 0=False, 1=True
            - uncertain_vars: List of uncertain variable indices
        """
        self.consensus_assignment = []
        uncertain_vars = []

        for var_idx in range(self.instance.num_vars):
            support = self.variable_support[var_idx]

            if support >= self.consensus_threshold_high:
                self.consensus_assignment.append(1)  # True
            elif support <= self.consensus_threshold_low:
                self.consensus_assignment.append(0)  # False
            else:
                self.consensus_assignment.append(-1)  # Uncertain
                uncertain_vars.append(var_idx)

        num_certain = sum(1 for x in self.consensus_assignment if x != -1)
        if self.verbose:
            print(f"Consensus: {num_certain}/{self.instance.num_vars} variables certain, "
                  f"{len(uncertain_vars)} uncertain")

        return self.consensus_assignment, uncertain_vars

    def create_partial_solution(self) -> SATSolution:
        """Create solution from consensus assignment."""
        assignment = []
        for val in self.consensus_assignment:
            if val == 1:
                assignment.append(True)
            elif val == 0:
                assignment.append(False)
            else:
                # Random for uncertain
                assignment.append(random.choice([True, False]))

        return SATSolution(self.instance.num_vars, assignment)

    def walksat_repair(self, solution: SATSolution, max_flips: int = 1000) -> SATSolution:
        """
        Repair solution using WalkSAT-style local search.

        Args:
            solution: Solution to repair
            max_flips: Maximum number of flips

        Returns:
            Repaired solution
        """
        evaluator = SATEvaluator(self.instance, use_weighted=False)
        repaired = solution.copy()

        for flip_count in range(max_flips):
            # Check which clauses are unsatisfied
            _, unsatisfied = evaluator.count_satisfied_clauses(repaired)

            if len(unsatisfied) == 0:
                break

            # Pick a random unsatisfied clause
            unsat_idx = random.choice(list(unsatisfied))
            unsat_clause = self.instance.clauses[unsat_idx]

            # Find best variable to flip in this clause
            best_var = None
            best_break_count = float('inf')

            for literal in unsat_clause:
                var_idx = abs(literal) - 1

                # Count how many currently satisfied clauses would break
                repaired.flip(var_idx)
                _, new_unsatisfied = evaluator.count_satisfied_clauses(repaired)
                break_count = len(new_unsatisfied)
                repaired.flip(var_idx)  # Flip back

                if break_count < best_break_count:
                    best_break_count = break_count
                    best_var = var_idx

            # Flip the best variable
            if best_var is not None:
                repaired.flip(best_var)

        return repaired

    def consensus_guided_restart(self, uncertain_vars: List[int]) -> SATSolution:
        """
        Run GA on reduced problem with consensus variables fixed.

        Args:
            uncertain_vars: List of uncertain variable indices

        Returns:
            Best solution found
        """
        if self.verbose:
            print(f"\nConsensus-guided restart on {len(uncertain_vars)} uncertain variables...")

        # Create a smaller GA focused on uncertain variables
        # For simplicity, run standard GA with consensus-biased initialization
        ga = GeneticAlgorithm(
            self.instance,
            population_size=50,
            elite_size=2,
            max_generations=50,
            stall_generations=20,
            verbose=False
        )

        # Override initialization to use consensus
        original_init = ga.initialize_population

        def consensus_init():
            ga.population = []
            for _ in range(ga.population_size):
                sol = self.create_partial_solution()
                # Add some variation to uncertain variables
                for var_idx in uncertain_vars:
                    if random.random() < 0.3:
                        sol.flip(var_idx)
                ga.population.append(sol)

        ga.initialize_population = consensus_init

        # Run GA
        best_sol, best_fit, is_sat = ga.run()

        return best_sol

    def break_ties_with_bagging(self, uncertain_vars: List[int]) -> Dict[int, float]:
        """
        Break ties using clause subset bagging.

        Args:
            uncertain_vars: List of uncertain variable indices

        Returns:
            Updated support for uncertain variables
        """
        if self.verbose:
            print(f"Breaking ties with bagging for {len(uncertain_vars)} variables...")

        num_bags = 10
        bag_size = max(1, int(0.7 * self.instance.num_clauses))
        updated_support = {}

        for var_idx in uncertain_vars:
            support_sum = 0.0

            for bag in range(num_bags):
                # Sample clauses
                sampled_clauses = random.sample(range(self.instance.num_clauses), bag_size)

                # Count support in sampled solutions that satisfy these clauses well
                bag_support = 0.0
                bag_weight = 0.0

                for sol_idx, solution in enumerate(self.all_solutions):
                    # Evaluate on sampled clauses
                    satisfied = 0
                    for clause_idx in sampled_clauses:
                        clause = self.instance.clauses[clause_idx]
                        if any(solution.get_value(lit) for lit in clause):
                            satisfied += 1

                    fitness = satisfied / bag_size
                    weight = np.exp(self.beta * fitness)

                    if solution.assignment[var_idx]:
                        bag_support += weight
                    bag_weight += weight

                if bag_weight > 0:
                    support_sum += bag_support / bag_weight

            updated_support[var_idx] = support_sum / num_bags

        return updated_support

    def iterative_woc(self) -> Tuple[SATSolution, float, bool]:
        """
        Main iterative WoC algorithm.

        Returns:
            Tuple of (best_solution, best_fitness, is_satisfying)
        """
        evaluator = SATEvaluator(self.instance, use_weighted=False)

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"WoC Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'='*70}")

            # Step 1: Build expert pool (first iteration)
            if iteration == 0:
                self.build_expert_pool()

            # Step 2: Extract opinions from experts
            self.extract_opinions()

            # Step 3: Compute variable support
            self.compute_variable_support()

            # Step 4: Extract consensus assignment
            consensus, uncertain_vars = self.extract_consensus_assignment()

            # Step 5: Create solution from consensus
            consensus_sol = self.create_partial_solution()

            # Step 6: Evaluate consensus solution
            cons_fit, cons_flips, cons_sat = evaluator.evaluate(consensus_sol)

            if self.verbose:
                print(f"Consensus solution: fitness={cons_fit:.4f}, satisfying={cons_sat}")

            # Update best
            if cons_fit > self.best_fitness:
                self.best_solution = consensus_sol.copy()
                self.best_fitness = cons_fit

            # Step 7: Check if solved
            if cons_sat:
                if self.verbose:
                    print(f"\n*** SOLUTION FOUND in iteration {iteration + 1} ***")
                return self.best_solution, self.best_fitness, True

            # Step 8: Repair with WalkSAT
            if self.verbose:
                print(f"Attempting WalkSAT repair...")

            repaired_sol = self.walksat_repair(consensus_sol)
            rep_fit, rep_flips, rep_sat = evaluator.evaluate(repaired_sol)

            if self.verbose:
                print(f"Repaired solution: fitness={rep_fit:.4f}, satisfying={rep_sat}")

            # Update best
            if rep_fit > self.best_fitness:
                self.best_solution = repaired_sol.copy()
                self.best_fitness = rep_fit

            if rep_sat:
                if self.verbose:
                    print(f"\n*** SOLUTION FOUND after repair in iteration {iteration + 1} ***")
                return self.best_solution, self.best_fitness, True

            # Step 9: Consensus-guided restart if many uncertain variables
            if len(uncertain_vars) > 0.1 * self.instance.num_vars:
                restart_sol = self.consensus_guided_restart(uncertain_vars)
                rst_fit, rst_flips, rst_sat = evaluator.evaluate(restart_sol)

                if self.verbose:
                    print(f"Restart solution: fitness={rst_fit:.4f}, satisfying={rst_sat}")

                if rst_fit > self.best_fitness:
                    self.best_solution = restart_sol.copy()
                    self.best_fitness = rst_fit

                if rst_sat:
                    if self.verbose:
                        print(f"\n*** SOLUTION FOUND after restart in iteration {iteration + 1} ***")
                    return self.best_solution, self.best_fitness, True

                # Add restart solution back to pool
                self.all_solutions.append(restart_sol)
                self.all_fitness.append(rst_fit)

            # Step 10: Break ties with bagging if needed
            if len(uncertain_vars) > 0 and iteration < self.max_iterations - 1:
                updated_support = self.break_ties_with_bagging(uncertain_vars)

                # Update variable support for uncertain variables
                for var_idx, new_support in updated_support.items():
                    self.variable_support[var_idx] = new_support

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"WoC completed: Best fitness = {self.best_fitness:.4f}")
            print(f"{'='*70}")

        final_fit, final_flips, is_sat = evaluator.evaluate(self.best_solution)
        return self.best_solution, final_fit, is_sat


if __name__ == "__main__":
    print("Testing Wisdom of Crowds (WoC) for 3-SAT\n")
    print("=" * 70)

    # Test with a simple instance
    print("\n[Test 1] Testing WoC on simple 3-SAT instance...")
    from solve_and_eval import read_dimacs_cnf
    import os

    test_clauses = [
        [1, 2, 3],
        [-1, 2, -3],
        [1, -2, 3],
        [-1, -2, -3],
        [1, 2, -3],
        [1, -2, -3],
        [-1, 2, 3]
    ]
    test_instance = SATInstance(3, test_clauses)

    woc = WisdomOfCrowds(
        test_instance,
        num_experts=8,
        top_k=3,
        max_iterations=2,
        parallel=False,  # Use serial for testing
        verbose=True
    )

    best_sol, best_fit, is_sat = woc.iterative_woc()
    print(f"\nFinal result:")
    print(f"  Best solution: {best_sol.to_string()}")
    print(f"  Fitness: {best_fit:.4f}")
    print(f"  Satisfying: {is_sat}")

    # Test with generated file if available
    print("\n" + "=" * 70)
    print("\n[Test 2] Testing with generated CNF file...")
    try:
        test_cnf_path = "data/test/test_3partition_easy.cnf"
        if os.path.exists(test_cnf_path):
            instance_from_file = read_dimacs_cnf(test_cnf_path)
            print(f"Loaded instance: {instance_from_file.num_vars} vars, "
                  f"{instance_from_file.num_clauses} clauses")

            woc_file = WisdomOfCrowds(
                instance_from_file,
                num_experts=16,
                top_k=5,
                max_iterations=2,
                parallel=False,
                verbose=True
            )

            best_sol_file, best_fit_file, is_sat_file = woc_file.iterative_woc()
            print(f"\nFinal result:")
            print(f"  Best fitness: {best_fit_file:.4f}")
            print(f"  Satisfying: {is_sat_file}")
        else:
            print(f"CNF file not found: {test_cnf_path}")
            print("(Run data_generation.py first)")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("WoC testing completed!")

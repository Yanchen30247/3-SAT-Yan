"""
Solution Representation and Evaluation Functions for 3-SAT Problem

This module provides encoding, fitness evaluation, and adaptive clause weighting
for solving 3-SAT problems using genetic algorithms or local search.
"""

import random
from typing import List, Tuple, Set
import numpy as np


class SATInstance:
    """Represents a 3-SAT problem instance."""

    def __init__(self, num_vars: int, clauses: List[List[int]]):
        """
        Initialize a SAT instance.

        Args:
            num_vars: Number of variables
            clauses: List of clauses, each clause is a list of literals
        """
        self.num_vars = num_vars
        self.clauses = clauses
        self.num_clauses = len(clauses)

        # Initialize clause weights (for dynamic weighting)
        self.clause_weights = np.ones(self.num_clauses, dtype=float)

        # Track unsatisfied clause history
        self.unsatisfied_history = [set() for _ in range(self.num_clauses)]

    def reset_weights(self):
        """Reset all clause weights to 1.0."""
        self.clause_weights = np.ones(self.num_clauses, dtype=float)

    def update_clause_weights(self, unsatisfied_clauses: Set[int],
                             increase_factor: float = 1.1,
                             decay_factor: float = 0.99):
        """
        Update clause weights based on which clauses are unsatisfied.

        Args:
            unsatisfied_clauses: Set of indices of unsatisfied clauses
            increase_factor: Factor to increase weight of unsatisfied clauses
            decay_factor: Factor to decay all weights (prevent unbounded growth)
        """
        # Increase weights for unsatisfied clauses
        for idx in unsatisfied_clauses:
            self.clause_weights[idx] *= increase_factor

        # Optional: decay all weights to prevent unbounded growth
        self.clause_weights *= decay_factor

        # Normalize to prevent numerical issues
        max_weight = np.max(self.clause_weights)
        if max_weight > 100:
            self.clause_weights /= (max_weight / 100)


class SATSolution:
    """
    Represents a solution (assignment) to a SAT problem.

    Encoding: A bitstring of length n where each bit represents a variable assignment.
    - assignment[i] = True means variable (i+1) is True
    - assignment[i] = False means variable (i+1) is False
    """

    def __init__(self, num_vars: int, assignment: List[bool] = None):
        """
        Initialize a SAT solution.

        Args:
            num_vars: Number of variables
            assignment: Initial assignment (random if None)
        """
        self.num_vars = num_vars
        if assignment is None:
            self.assignment = [random.choice([True, False]) for _ in range(num_vars)]
        else:
            self.assignment = assignment.copy()

    def get_value(self, literal: int) -> bool:
        """
        Get the truth value of a literal.

        Args:
            literal: A literal (positive or negative integer)

        Returns:
            Truth value of the literal
        """
        var_index = abs(literal) - 1
        var_value = self.assignment[var_index]
        return var_value if literal > 0 else not var_value

    def flip(self, var_index: int):
        """Flip the value of a variable."""
        self.assignment[var_index] = not self.assignment[var_index]

    def copy(self):
        """Create a copy of this solution."""
        return SATSolution(self.num_vars, self.assignment)

    def to_string(self) -> str:
        """Convert assignment to string representation."""
        return ''.join(['1' if x else '0' for x in self.assignment])

    @staticmethod
    def from_string(bitstring: str):
        """Create solution from string representation."""
        assignment = [c == '1' for c in bitstring]
        return SATSolution(len(assignment), assignment)


class SATEvaluator:
    """Evaluates SAT solutions with adaptive fitness and clause weighting."""

    def __init__(self, instance: SATInstance,
                 use_weighted: bool = True,
                 use_flip_distance: bool = True):
        """
        Initialize evaluator.

        Args:
            instance: SAT instance to evaluate against
            use_weighted: Whether to use weighted clause satisfaction
            use_flip_distance: Whether to use minimum flip distance for tie-breaking
        """
        self.instance = instance
        self.use_weighted = use_weighted
        self.use_flip_distance = use_flip_distance

        # Statistics
        self.num_evaluations = 0

    def is_clause_satisfied(self, solution: SATSolution, clause: List[int]) -> bool:
        """
        Check if a clause is satisfied by the solution.

        Args:
            solution: Solution to evaluate
            clause: Clause to check

        Returns:
            True if clause is satisfied, False otherwise
        """
        for literal in clause:
            if solution.get_value(literal):
                return True
        return False

    def count_satisfied_clauses(self, solution: SATSolution) -> Tuple[int, Set[int]]:
        """
        Count number of satisfied clauses.

        Args:
            solution: Solution to evaluate

        Returns:
            Tuple of (number of satisfied clauses, set of unsatisfied clause indices)
        """
        satisfied_count = 0
        unsatisfied_indices = set()

        for idx, clause in enumerate(self.instance.clauses):
            if self.is_clause_satisfied(solution, clause):
                satisfied_count += 1
            else:
                unsatisfied_indices.add(idx)

        return satisfied_count, unsatisfied_indices

    def compute_basic_fitness(self, solution: SATSolution) -> Tuple[float, Set[int]]:
        """
        Compute basic fitness: normalized satisfied clauses S/m.

        Args:
            solution: Solution to evaluate

        Returns:
            Tuple of (normalized fitness [0,1], set of unsatisfied clause indices)
        """
        satisfied_count, unsatisfied = self.count_satisfied_clauses(solution)
        normalized_fitness = satisfied_count / self.instance.num_clauses
        return normalized_fitness, unsatisfied

    def compute_weighted_fitness(self, solution: SATSolution) -> Tuple[float, Set[int]]:
        """
        Compute weighted fitness: F(x) = Σ w_c · 1[c is satisfied].

        Args:
            solution: Solution to evaluate

        Returns:
            Tuple of (weighted fitness, set of unsatisfied clause indices)
        """
        weighted_sum = 0.0
        unsatisfied_indices = set()

        for idx, clause in enumerate(self.instance.clauses):
            if self.is_clause_satisfied(solution, clause):
                weighted_sum += self.instance.clause_weights[idx]
            else:
                unsatisfied_indices.add(idx)

        # Normalize by total weight for comparison
        total_weight = np.sum(self.instance.clause_weights)
        normalized_fitness = weighted_sum / total_weight if total_weight > 0 else 0.0

        return normalized_fitness, unsatisfied_indices

    def estimate_min_flips(self, solution: SATSolution,
                          unsatisfied_clauses: Set[int]) -> int:
        """
        Estimate minimum number of flips needed to satisfy all unsatisfied clauses.

        This is a heuristic for tie-breaking: count how many variables appear
        most frequently in unsatisfied clauses.

        Args:
            solution: Current solution
            unsatisfied_clauses: Set of unsatisfied clause indices

        Returns:
            Estimated minimum flips (lower is better)
        """
        if not unsatisfied_clauses:
            return 0

        # Count variable frequency in unsatisfied clauses
        var_frequency = np.zeros(self.instance.num_vars, dtype=int)

        for idx in unsatisfied_clauses:
            clause = self.instance.clauses[idx]
            for literal in clause:
                var_index = abs(literal) - 1
                # Check if flipping this variable would help
                if not solution.get_value(literal):
                    var_frequency[var_index] += 1

        # Greedy estimate: sort by frequency and estimate coverage
        sorted_freq = sorted(var_frequency, reverse=True)

        # Simple heuristic: sum of top frequencies
        # Better heuristic would do set cover approximation
        flip_estimate = len(unsatisfied_clauses)
        if len(sorted_freq) > 0 and sorted_freq[0] > 0:
            # Roughly estimate based on coverage
            flip_estimate = max(1, len(unsatisfied_clauses) // max(1, sorted_freq[0]))

        return flip_estimate

    def evaluate(self, solution: SATSolution) -> Tuple[float, int, bool]:
        """
        Evaluate a solution with comprehensive fitness.

        Returns:
            Tuple of (fitness, flip_distance, is_satisfying)
            - fitness: Primary fitness score (higher is better)
            - flip_distance: Estimated minimum flips for tie-breaking (lower is better)
            - is_satisfying: Whether solution satisfies all clauses
        """
        self.num_evaluations += 1

        # Compute fitness based on mode
        if self.use_weighted:
            fitness, unsatisfied = self.compute_weighted_fitness(solution)
        else:
            fitness, unsatisfied = self.compute_basic_fitness(solution)

        # Check if solution is satisfying
        is_satisfying = len(unsatisfied) == 0

        # Compute flip distance for tie-breaking
        if self.use_flip_distance:
            flip_distance = self.estimate_min_flips(solution, unsatisfied)
        else:
            flip_distance = len(unsatisfied)

        return fitness, flip_distance, is_satisfying

    def compare_solutions(self, sol1: SATSolution, sol2: SATSolution) -> int:
        """
        Compare two solutions.

        Args:
            sol1: First solution
            sol2: Second solution

        Returns:
            1 if sol1 is better, -1 if sol2 is better, 0 if equal
        """
        fit1, flip1, sat1 = self.evaluate(sol1)
        fit2, flip2, sat2 = self.evaluate(sol2)

        # Primary: fitness (higher is better)
        if fit1 > fit2:
            return 1
        elif fit1 < fit2:
            return -1

        # Secondary: flip distance (lower is better)
        if flip1 < flip2:
            return 1
        elif flip1 > flip2:
            return -1

        return 0

    def get_best_flip(self, solution: SATSolution) -> Tuple[int, float]:
        """
        Find the best variable to flip using a greedy heuristic.

        Args:
            solution: Current solution

        Returns:
            Tuple of (best variable index to flip, fitness gain)
        """
        best_var = -1
        best_gain = -float('inf')
        current_fitness, _, _ = self.evaluate(solution)

        for var_idx in range(self.instance.num_vars):
            # Flip and evaluate
            solution.flip(var_idx)
            new_fitness, _, _ = self.evaluate(solution)
            gain = new_fitness - current_fitness

            if gain > best_gain:
                best_gain = gain
                best_var = var_idx

            # Flip back
            solution.flip(var_idx)

        return best_var, best_gain


def read_dimacs_cnf(filepath: str) -> SATInstance:
    """
    Read a DIMACS CNF file and create a SAT instance.

    Args:
        filepath: Path to DIMACS CNF file

    Returns:
        SATInstance object
    """
    num_vars = 0
    clauses = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments
            if line.startswith('c'):
                continue

            # Parse problem line
            if line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
                continue

            # Parse clause
            if line and not line.startswith('c') and not line.startswith('p'):
                literals = [int(x) for x in line.split() if int(x) != 0]
                if literals:
                    clauses.append(literals)

    return SATInstance(num_vars, clauses)


if __name__ == "__main__":
    print("Testing SAT Solution Representation and Evaluation\n")
    print("=" * 70)

    # Test 1: Create a simple SAT instance
    print("\n[Test 1] Creating a simple 3-SAT instance...")
    # Example: (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ ¬x3) ∧ (x1 ∨ ¬x2 ∨ x3)
    test_clauses = [
        [1, 2, 3],      # x1 OR x2 OR x3
        [-1, 2, -3],    # NOT x1 OR x2 OR NOT x3
        [1, -2, 3]      # x1 OR NOT x2 OR x3
    ]
    instance = SATInstance(3, test_clauses)
    print(f"  Variables: {instance.num_vars}")
    print(f"  Clauses: {instance.num_clauses}")
    print(f"  Clause weights: {instance.clause_weights}")

    # Test 2: Create and evaluate solutions
    print("\n[Test 2] Creating and evaluating solutions...")

    # Solution 1: All True
    sol1 = SATSolution(3, [True, True, True])
    print(f"\n  Solution 1: {sol1.to_string()}")

    evaluator = SATEvaluator(instance, use_weighted=False)
    fitness1, flip1, sat1 = evaluator.evaluate(sol1)
    print(f"    Fitness: {fitness1:.3f}")
    print(f"    Min flips estimate: {flip1}")
    print(f"    Is satisfying: {sat1}")

    # Solution 2: All False
    sol2 = SATSolution(3, [False, False, False])
    print(f"\n  Solution 2: {sol2.to_string()}")
    fitness2, flip2, sat2 = evaluator.evaluate(sol2)
    print(f"    Fitness: {fitness2:.3f}")
    print(f"    Min flips estimate: {flip2}")
    print(f"    Is satisfying: {sat2}")

    # Solution 3: Mixed
    sol3 = SATSolution(3, [True, False, True])
    print(f"\n  Solution 3: {sol3.to_string()}")
    fitness3, flip3, sat3 = evaluator.evaluate(sol3)
    print(f"    Fitness: {fitness3:.3f}")
    print(f"    Min flips estimate: {flip3}")
    print(f"    Is satisfying: {sat3}")

    # Test 3: Weighted evaluation
    print("\n[Test 3] Testing weighted fitness evaluation...")
    evaluator_weighted = SATEvaluator(instance, use_weighted=True)

    # Update weights to favor unsatisfied clauses
    _, unsatisfied = evaluator_weighted.count_satisfied_clauses(sol2)
    instance.update_clause_weights(unsatisfied, increase_factor=2.0)
    print(f"  Updated clause weights: {instance.clause_weights}")

    fitness_weighted, _, _ = evaluator_weighted.evaluate(sol2)
    print(f"  Weighted fitness for solution 2: {fitness_weighted:.3f}")

    # Test 4: Solution comparison
    print("\n[Test 4] Comparing solutions...")
    result = evaluator.compare_solutions(sol1, sol2)
    if result > 0:
        print(f"  Solution 1 is better than Solution 2")
    elif result < 0:
        print(f"  Solution 2 is better than Solution 1")
    else:
        print(f"  Solutions are equivalent")

    # Test 5: Find best flip
    print("\n[Test 5] Finding best variable to flip...")
    best_var, best_gain = evaluator.get_best_flip(sol2)
    print(f"  Best variable to flip: {best_var + 1}")
    print(f"  Expected fitness gain: {best_gain:.3f}")

    # Test flip
    sol2.flip(best_var)
    new_fitness, new_flip, new_sat = evaluator.evaluate(sol2)
    print(f"  After flipping: fitness = {new_fitness:.3f}, satisfying = {new_sat}")

    # Test 6: Random solution generation and evaluation
    print("\n[Test 6] Generating and evaluating random solutions...")
    num_random = 5
    best_solution = None
    best_fitness = -1

    for i in range(num_random):
        random_sol = SATSolution(3)
        fit, flips, is_sat = evaluator.evaluate(random_sol)
        print(f"  Random solution {i+1}: {random_sol.to_string()} -> " +
              f"fitness={fit:.3f}, flips={flips}, sat={is_sat}")

        if fit > best_fitness or (fit == best_fitness and best_solution is None):
            best_fitness = fit
            best_solution = random_sol

    print(f"\n  Best random solution: {best_solution.to_string()}")
    print(f"  Best fitness: {best_fitness:.3f}")

    # Test 7: Test with generated CNF file (if exists)
    print("\n[Test 7] Testing with generated CNF file...")
    try:
        import os
        test_cnf_path = "data/test/test_3partition_easy.cnf"
        if os.path.exists(test_cnf_path):
            instance_from_file = read_dimacs_cnf(test_cnf_path)
            print(f"  Loaded instance from {test_cnf_path}")
            print(f"  Variables: {instance_from_file.num_vars}")
            print(f"  Clauses: {instance_from_file.num_clauses}")

            # Evaluate a random solution
            eval_file = SATEvaluator(instance_from_file, use_weighted=True)
            random_solution = SATSolution(instance_from_file.num_vars)
            fit, flips, is_sat = eval_file.evaluate(random_solution)
            print(f"  Random solution fitness: {fit:.3f}")
            print(f"  Is satisfying: {is_sat}")
            print(f"  Total evaluations: {eval_file.num_evaluations}")
        else:
            print(f"  CNF file not found: {test_cnf_path}")
            print(f"  (Run data_generation.py first to generate test files)")
    except Exception as e:
        print(f"  Error loading CNF file: {e}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print(f"Total evaluations performed: {evaluator.num_evaluations}")

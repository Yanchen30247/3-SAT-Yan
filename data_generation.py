"""
Data Generation for 3-Partition Problem using DIMACS CNF Format

This module generates test instances for the 3-partition problem encoded as
3-SAT CNF formulas in DIMACS format.
"""

import random
import os
from typing import List, Tuple, Dict
import json
from datetime import datetime


class ThreePartitionCNFGenerator:
    """Generator for 3-partition problem instances in DIMACS CNF format."""

    def __init__(self, seed: int = None):
        """
        Initialize the generator with an optional random seed.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        self.seed = seed

    def generate_3partition_instance(self, m: int) -> Tuple[List[int], int]:
        """
        Generate a 3-partition problem instance with guaranteed solution.

        Args:
            m: Number of triplets (n = 3m elements total)

        Returns:
            Tuple of (multiset S, target sum T)
        """
        n = 3 * m
        # Target sum T - choose a reasonable value
        T = random.randint(m * 30, m * 60)

        # Generate satisfiable instance by constructing valid triplets
        S = []
        for _ in range(m):
            # Generate three numbers that sum to T
            # Ensure each is strictly between T/4 and T/2
            lower_bound = T // 4 + 1
            upper_bound = T // 2 - 1

            # First two numbers
            a = random.randint(lower_bound, upper_bound)
            # Second number must allow room for third
            max_b = min(upper_bound, T - a - lower_bound)
            min_b = max(lower_bound, T - a - upper_bound)

            if min_b > max_b:
                # Adjust if constraints are violated
                a = T // 3

            b = random.randint(max(lower_bound, T - a - upper_bound),
                             min(upper_bound, T - a - lower_bound))
            c = T - a - b

            # Ensure c is in valid range
            if c < lower_bound or c > upper_bound:
                # Use balanced distribution
                a = T // 3 + random.randint(-T//12, T//12)
                b = T // 3 + random.randint(-T//12, T//12)
                c = T - a - b

            S.extend([a, b, c])

        # Shuffle to hide the structure
        random.shuffle(S)
        return S, T

    def encode_3partition_to_cnf(self, S: List[int], T: int, m: int) -> Tuple[List[List[int]], int, Dict]:
        """
        Encode 3-partition problem to CNF formula.

        Variable encoding:
        - x_{i,j,k} means element i is in triplet j at position k
        - Variable number: (i * m * 3) + (j * 3) + k + 1

        Args:
            S: Multiset of integers
            T: Target sum
            m: Number of triplets

        Returns:
            Tuple of (clauses, num_variables, metadata)
        """
        n = len(S)
        clauses = []

        def var(i: int, j: int, k: int) -> int:
            """Convert (element, triplet, position) to variable number."""
            return (i * m * 3) + (j * 3) + k + 1

        num_vars = n * m * 3

        # Constraint 1: Each element must be in exactly one triplet at one position
        for i in range(n):
            # At least one assignment
            clause = []
            for j in range(m):
                for k in range(3):
                    clause.append(var(i, j, k))
            clauses.append(clause)

            # At most one assignment (pairwise exclusion)
            for j1 in range(m):
                for k1 in range(3):
                    for j2 in range(m):
                        for k2 in range(3):
                            if (j1, k1) < (j2, k2):
                                clauses.append([-var(i, j1, k1), -var(i, j2, k2)])

        # Constraint 2: Each position in each triplet must have exactly one element
        for j in range(m):
            for k in range(3):
                # At least one element
                clause = [var(i, j, k) for i in range(n)]
                clauses.append(clause)

                # At most one element
                for i1 in range(n):
                    for i2 in range(i1 + 1, n):
                        clauses.append([-var(i1, j, k), -var(i2, j, k)])

        metadata = {
            'n': n,
            'm': m,
            'T': T,
            'S': S,
            'encoding': 'direct',
            'num_variables': num_vars,
            'num_clauses': len(clauses)
        }

        return clauses, num_vars, metadata

    def generate_satisfiable_cnf(self, m: int, difficulty: str = "medium") -> Tuple[List[List[int]], int, Dict]:
        """
        Generate a satisfiable 3-partition CNF instance.

        Args:
            m: Number of triplets
            difficulty: "easy", "medium", or "hard"

        Returns:
            Tuple of (clauses, num_variables, metadata)
        """
        S, T = self.generate_3partition_instance(m)
        clauses, num_vars, metadata = self.encode_3partition_to_cnf(S, T, m)

        # Add difficulty modifiers
        if difficulty == "easy":
            # Add redundant helpful clauses
            pass
        elif difficulty == "hard":
            # Add misleading neutral clauses
            extra_clauses = self._add_neutral_clauses(clauses, num_vars, ratio=0.1)
            clauses.extend(extra_clauses)

        metadata['difficulty'] = difficulty
        metadata['satisfiable'] = True
        metadata['generation_method'] = 'constructive'

        return clauses, num_vars, metadata

    def generate_random_3sat(self, n: int, clause_ratio: float = 4.3) -> Tuple[List[List[int]], int, Dict]:
        """
        Generate random 3-SAT instance (potentially unsatisfiable).

        Args:
            n: Number of variables
            clause_ratio: Ratio of clauses to variables (m/n)

        Returns:
            Tuple of (clauses, num_variables, metadata)
        """
        m = int(n * clause_ratio)
        clauses = []

        for _ in range(m):
            # Randomly select 3 distinct variables
            vars_selected = random.sample(range(1, n + 1), 3)
            # Randomly assign polarities
            clause = [v if random.random() > 0.5 else -v for v in vars_selected]
            clauses.append(clause)

        metadata = {
            'n': n,
            'm': m,
            'clause_ratio': clause_ratio,
            'satisfiable': 'unknown',
            'generation_method': 'random',
            'difficulty': 'challenging'
        }

        return clauses, n, metadata

    def generate_challenging_set(self, n: int, clause_ratio: float = 4.3,
                                target_difficulty: str = "hard") -> Tuple[List[List[int]], int, Dict]:
        """
        Generate challenging/potentially unsatisfiable 3-SAT instance.

        Uses random generation with controlled clause-to-variable ratio
        near the satisfiability threshold.

        Args:
            n: Number of variables
            clause_ratio: Ratio of clauses to variables (default ~4.3)
            target_difficulty: Difficulty target

        Returns:
            Tuple of (clauses, num_variables, metadata)
        """
        # Add variation to clause ratio
        actual_ratio = clause_ratio + random.uniform(-0.2, 0.2)
        clauses, num_vars, metadata = self.generate_random_3sat(n, actual_ratio)

        metadata['target_difficulty'] = target_difficulty
        metadata['actual_clause_ratio'] = actual_ratio

        return clauses, num_vars, metadata

    def _add_neutral_clauses(self, clauses: List[List[int]], num_vars: int,
                            ratio: float = 0.1) -> List[List[int]]:
        """
        Add neutral clauses that don't provide obvious hints.

        Args:
            clauses: Existing clauses
            num_vars: Number of variables
            ratio: Ratio of neutral clauses to add

        Returns:
            List of neutral clauses
        """
        num_neutral = int(len(clauses) * ratio)
        neutral_clauses = []

        for _ in range(num_neutral):
            vars_selected = random.sample(range(1, num_vars + 1), 3)
            clause = [v if random.random() > 0.5 else -v for v in vars_selected]
            neutral_clauses.append(clause)

        return neutral_clauses

    def save_dimacs_cnf(self, clauses: List[List[int]], num_vars: int,
                       filepath: str, metadata: Dict = None):
        """
        Save CNF formula in DIMACS format.

        Format:
        - First line: p cnf n m (n variables, m clauses)
        - Each clause: space-separated literals ending with 0
        - Comments start with 'c'

        Args:
            clauses: List of clauses (each clause is list of literals)
            num_vars: Number of variables
            filepath: Output file path
            metadata: Optional metadata to include as comments
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            # Write header comments
            f.write(f"c Generated by ThreePartitionCNFGenerator\n")
            f.write(f"c Date: {datetime.now().isoformat()}\n")
            if self.seed is not None:
                f.write(f"c Seed: {self.seed}\n")

            if metadata:
                for key, value in metadata.items():
                    f.write(f"c {key}: {value}\n")

            # Write problem line
            f.write(f"p cnf {num_vars} {len(clauses)}\n")

            # Write clauses
            for clause in clauses:
                f.write(' '.join(map(str, clause)) + ' 0\n')

    def generate_dataset(self, output_dir: str = "data",
                        sizes: List[int] = None,
                        difficulty_levels: List[str] = None,
                        instances_per_config: int = 5):
        """
        Generate a complete dataset with multiple configurations.

        Args:
            output_dir: Base output directory
            sizes: List of problem sizes (m values for 3-partition)
            difficulty_levels: List of difficulty levels
            instances_per_config: Number of instances per configuration
        """
        if sizes is None:
            sizes = [5, 10, 15, 20]
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium", "hard"]

        summary = {
            'generation_date': datetime.now().isoformat(),
            'seed': self.seed,
            'configurations': []
        }

        for size in sizes:
            for difficulty in difficulty_levels:
                for instance_id in range(instances_per_config):
                    # Set seed for this instance
                    instance_seed = (self.seed or 0) + instance_id
                    random.seed(instance_seed)

                    # Generate satisfiable instance
                    clauses, num_vars, metadata = self.generate_satisfiable_cnf(size, difficulty)

                    # Create filename
                    n = 3 * size
                    ratio = len(clauses) / num_vars if num_vars > 0 else 0
                    subdir = f"{n}_{ratio:.1f}"
                    filename = f"{instance_seed}_{difficulty}_sat.cnf"
                    filepath = os.path.join(output_dir, subdir, filename)

                    # Save instance
                    self.save_dimacs_cnf(clauses, num_vars, filepath, metadata)

                    # Record in summary
                    summary['configurations'].append({
                        'file': filepath,
                        'size': size,
                        'n': n,
                        'difficulty': difficulty,
                        'seed': instance_seed,
                        'satisfiable': True,
                        'num_variables': num_vars,
                        'num_clauses': len(clauses),
                        'clause_ratio': ratio
                    })

                    print(f"Generated: {filepath}")

        # Generate challenging/unsatisfiable instances
        for n in [s * 3 for s in sizes]:
            for instance_id in range(instances_per_config):
                instance_seed = (self.seed or 0) + 1000 + instance_id
                random.seed(instance_seed)

                clauses, num_vars, metadata = self.generate_challenging_set(n, clause_ratio=4.3)

                ratio = len(clauses) / num_vars if num_vars > 0 else 0
                subdir = f"{n}_{ratio:.1f}"
                filename = f"{instance_seed}_challenging.cnf"
                filepath = os.path.join(output_dir, subdir, filename)

                self.save_dimacs_cnf(clauses, num_vars, filepath, metadata)

                summary['configurations'].append({
                    'file': filepath,
                    'n': n,
                    'difficulty': 'challenging',
                    'seed': instance_seed,
                    'satisfiable': 'unknown',
                    'num_variables': num_vars,
                    'num_clauses': len(clauses),
                    'clause_ratio': ratio
                })

                print(f"Generated: {filepath}")

        # Save summary
        summary_path = os.path.join(output_dir, "README.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Create human-readable README
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("3-Partition Problem Dataset in DIMACS CNF Format\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generation Date: {summary['generation_date']}\n")
            f.write(f"Random Seed: {summary['seed']}\n\n")
            f.write(f"Total Instances: {len(summary['configurations'])}\n\n")
            f.write("Directory Structure: data/{n}_{ratio}/{seed}_{tag}.cnf\n\n")
            f.write("Difficulty Levels:\n")
            f.write("  - easy: Straightforward instances\n")
            f.write("  - medium: Moderate difficulty\n")
            f.write("  - hard: Complex instances with neutral clauses\n")
            f.write("  - challenging: Random instances near SAT threshold\n\n")
            f.write("File Format: DIMACS CNF\n")
            f.write("  - First line: p cnf n m (n variables, m clauses)\n")
            f.write("  - Comments: lines starting with 'c'\n")
            f.write("  - Clauses: space-separated literals ending with 0\n")

        print(f"\nDataset generation complete!")
        print(f"Total instances: {len(summary['configurations'])}")
        print(f"Summary saved to: {summary_path}")
        print(f"README saved to: {readme_path}")

        return summary


if __name__ == "__main__":
    # Test data generation
    print("Testing 3-Partition CNF Data Generator\n")
    print("=" * 60)

    # Create generator with fixed seed for reproducibility
    generator = ThreePartitionCNFGenerator(seed=42)

    # Test 1: Generate a single satisfiable instance
    print("\n[Test 1] Generating single satisfiable instance (m=3, easy)...")
    clauses, num_vars, metadata = generator.generate_satisfiable_cnf(m=3, difficulty="easy")
    print(f"  Variables: {num_vars}")
    print(f"  Clauses: {len(clauses)}")
    print(f"  Metadata: {metadata}")

    # Save test instance
    test_file = "data/test/test_3partition_easy.cnf"
    generator.save_dimacs_cnf(clauses, num_vars, test_file, metadata)
    print(f"  Saved to: {test_file}")

    # Test 2: Generate a challenging instance
    print("\n[Test 2] Generating challenging instance (n=30)...")
    clauses, num_vars, metadata = generator.generate_challenging_set(n=30)
    print(f"  Variables: {num_vars}")
    print(f"  Clauses: {len(clauses)}")
    print(f"  Clause ratio: {len(clauses)/num_vars:.2f}")

    test_file2 = "data/test/test_challenging.cnf"
    generator.save_dimacs_cnf(clauses, num_vars, test_file2, metadata)
    print(f"  Saved to: {test_file2}")

    # Test 3: Generate full dataset
    print("\n[Test 3] Generating full dataset...")
    print("  Sizes: [2, 3, 5]")
    print("  Difficulty levels: [easy, medium, hard]")
    print("  Instances per config: 2")
    print()

    summary = generator.generate_dataset(
        output_dir="data",
        sizes=[2, 3, 5],
        difficulty_levels=["easy", "medium", "hard"],
        instances_per_config=2
    )

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print(f"Check the 'data/' directory for generated files.")

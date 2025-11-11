# 3-Partition Problem Solver with Genetic Algorithms and Wisdom of Crowds

## Project Overview

This project implements a comprehensive solver for the 3-partition problem using:
1. **Standard Genetic Algorithm (GA)** as a baseline
2. **Wisdom of Crowds (WoC)** approach for enhanced performance
3. **DIMACS CNF format** for data representation

## Files Structure

### 1. `data_generation.py`
Generates test instances for the 3-partition problem in DIMACS CNF format.

**Features:**
- Generates satisfiable instances with guaranteed solutions
- Creates challenging/unsatisfiable instances near SAT threshold
- Supports difficulty tiers (Easy/Medium/Hard)
- Produces structured dataset with metadata

**Key Functions:**
- `generate_3partition_instance()`: Creates valid 3-partition instances
- `encode_3partition_to_cnf()`: Encodes problem to CNF
- `generate_dataset()`: Batch generates test datasets

**Usage:**
```python
python data_generation.py
```

### 2. `solve_and_eval.py`
Solution representation and evaluation framework.

**Features:**
- Bitstring encoding (True/False assignments)
- Basic fitness: S/m (normalized satisfied clauses)
- Weighted fitness: F(x) = Σ w_c · 1[c satisfied]
- Dynamic clause weighting (SAPS-style)
- Minimum flip distance estimation for tie-breaking

**Key Classes:**
- `SATInstance`: Represents a SAT problem with clause weights
- `SATSolution`: Bitstring solution representation
- `SATEvaluator`: Comprehensive fitness evaluation with tie-breaking

**Usage:**
```python
from solve_and_eval import read_dimacs_cnf, SATEvaluator
instance = read_dimacs_cnf("path/to/file.cnf")
evaluator = SATEvaluator(instance)
fitness, flips, is_sat = evaluator.evaluate(solution)
```

### 3. `GA_main.py`
Standard Genetic Algorithm baseline implementation.

**Features:**
- **Initialization**: 50% random + 50% heuristic (unit clauses, pure literals, frequency-based)
- **Selection**: Tournament selection (k=2-4) with elitism (1-2 elite)
- **Crossover**: One-point, two-point, and uniform (with elite bias)
- **Mutation**: Directed mutation (0.5%-2%) prioritizing unsatisfied clause variables
- **Local Search**: GSAT-style hill climbing on elite (10-50 steps)
- **Diversity Restart**: After G_stall generations without improvement

**Key Parameters:**
- `population_size`: 50-200 individuals
- `elite_size`: 1-2 elite preserved
- `mutation_rate`: 0.005-0.02
- `stall_generations`: 20-50 (triggers restart)
- `local_search_steps`: 10-50

**Usage:**
```python
from GA_main import GeneticAlgorithm
ga = GeneticAlgorithm(instance, population_size=100, max_generations=1000)
best_sol, best_fit, is_sat = ga.run()
```

### 4. `WOC.py`
Wisdom of Crowds implementation.

**Features:**

#### Expert Pool Construction (16-64 experts)
- **Parameter Variation**: Population size, crossover/mutation rates, tournament size
- **Heuristic Switches**: Weighted fitness on/off, initialization strategy
- **Diversity**: Different restart thresholds, local search steps
- **Parallel/Serial Execution**: Supports both modes

#### Opinion Extraction
- Collects top-k (k=5) elite solutions from each expert
- Computes variable support: p_i = Σ_e Σ_x∈E_e w(x)·1[x_i=True] / Σ_e Σ_x∈E_e w(x)
- **Weighting**: w(x) = exp(β·zscore(F)) with diversity adjustment
- Reduces weight for similar solutions (deduplication)

#### Consensus Assignment & Skeleton Extraction
- **Voting Thresholds**:
  - p_i ≥ 0.7 → True
  - p_i ≤ 0.3 → False
  - 0.3 < p_i < 0.7 → Uncertain
- Fixes certain variables, runs GA on reduced problem (Consensus-Guided Restart)

#### Conflict & Tie Handling
- **WalkSAT Repair**: Fixes unsatisfied clauses by selecting variable flips that minimize breaks
- **Bagging**: Clause subset sampling to break ties on uncertain variables
- **Gradient-based Flip Estimation**: Uses flip benefit for final decisions

#### Iterative Refinement
- Adds repaired solutions back to expert pool
- Updates support p_i iteratively
- Converges or reaches budget limit

**Key Parameters:**
- `num_experts`: 16-64 expert GAs
- `top_k`: 5 elite solutions per expert
- `consensus_threshold_high`: 0.7 (for True)
- `consensus_threshold_low`: 0.3 (for False)
- `beta`: 2.0 (weight exponent)
- `max_iterations`: 3 WoC iterations

**Usage:**
```python
from WOC import WisdomOfCrowds
woc = WisdomOfCrowds(instance, num_experts=32, top_k=5, max_iterations=3)
best_sol, best_fit, is_sat = woc.iterative_woc()
```

## Algorithm Flow

### Standard GA Flow
```
1. Initialize population (50% random, 50% heuristic)
2. Evaluate fitness
3. While not converged:
   a. Tournament selection
   b. Crossover (one-point/two-point/uniform)
   c. Directed mutation
   d. Apply local search to elite
   e. Update clause weights
   f. Check for stall → diversity restart if needed
4. Return best solution
```

### WoC Flow
```
1. Build expert pool (16-64 diverse GA configurations)
2. For each iteration:
   a. Run all experts in parallel/serial
   b. Extract top-k elite from each expert
   c. Compute variable support with weighted voting
   d. Extract consensus assignment (certain/uncertain)
   e. Create solution from consensus
   f. If not satisfied:
      - Apply WalkSAT repair
      - If many uncertain vars → consensus-guided restart
      - Break ties with clause bagging
   g. Add repaired solutions back to pool
3. Return best solution found
```

## Data Format

### DIMACS CNF Format
```
c Comments start with 'c'
c Problem metadata
p cnf <num_vars> <num_clauses>
<lit1> <lit2> <lit3> 0
<lit1> <lit2> <lit3> 0
...
```

### Directory Structure
```
data/
  {n}_{ratio}/
    {seed}_{difficulty}.cnf
  README.json
  README.txt
```

## Experimental Setup

### Standard GA Baseline
- Population: 100
- Elite: 2
- Crossover: 0.8
- Mutation: 0.01
- Generations: 100
- Stall: 50

### WoC Configuration
- Experts: 32
- Top-k: 5
- Iterations: 3
- Thresholds: 0.7/0.3
- Parallel: True

## Performance Metrics

1. **Solution Quality**: Fitness score (satisfied clauses)
2. **Success Rate**: Percentage of instances solved
3. **Evaluations**: Total fitness evaluations
4. **Time**: Wall-clock time
5. **Convergence**: Generations to best solution

## Running Tests

### Generate Data
```bash
cd final_project
python data_generation.py
```

### Test Evaluation System
```bash
python solve_and_eval.py
```

### Test Standard GA
```bash
python GA_main.py
```

### Test WoC
```bash
python WOC.py
```

## Key Innovations

1. **Adaptive Clause Weighting**: Long-unsatisfied clauses get higher weights
2. **Directed Mutation**: Focuses on variables in unsatisfied clauses
3. **Hybrid Initialization**: Combines random and heuristic strategies
4. **Elite-Biased Crossover**: Uses elite patterns to guide offspring
5. **Diversity Restart**: Prevents premature convergence
6. **WoC Consensus**: Aggregates multiple expert opinions
7. **Skeleton Extraction**: Fixes certain variables, focuses on uncertain
8. **Conflict Repair**: WalkSAT for post-consensus refinement
9. **Iterative Refinement**: Feeds solutions back to improve consensus

## Future Extensions

- [ ] Implement more sophisticated local search (Novelty+)
- [ ] Add clause learning mechanisms
- [ ] Parallel island model GA
- [ ] Adaptive parameter control
- [ ] Integration with SAT solver backends
- [ ] Visualization of convergence and diversity
- [ ] Benchmark comparison with state-of-the-art solvers

## References

- GSAT/WalkSAT algorithms
- SAPS (Scaling and Probabilistic Smoothing)
- Wisdom of Crowds in optimization
- Genetic Algorithms for SAT

## Author

CSE545 Final Project
Date: November 10, 2025



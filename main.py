"""
Main Experimental Framework for 3-SAT Solver Comparison

This module provides:
- Experimental design: Standard GA vs GA+WoC comparison
- Metrics: Success rate, satisfaction, convergence time, stability
- Ablation studies: Component-wise contribution analysis
- Parameter sensitivity analysis
- GUI with visualization: Formula view, variable panel, evolution curves, structure graphs
"""

import sys
import os
import random
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # For GUI support
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import networkx as nx
from collections import defaultdict
from scipy import stats

# Import our modules
from solve_and_eval import SATInstance, SATSolution, SATEvaluator, read_dimacs_cnf
from GA_main import GeneticAlgorithm
from WOC import WisdomOfCrowds


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    algorithm: str
    instance_name: str
    n_vars: int
    n_clauses: int
    clause_ratio: float
    success: bool
    best_fitness: float
    satisfied_clauses: int
    generations: int
    evaluations: int
    time_seconds: float
    convergence_generation: int
    final_diversity: float = 0.0

    def to_dict(self):
        return asdict(self)


class ExperimentRunner:
    """Run controlled experiments comparing algorithms."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[ExperimentResult] = []

    def run_single_experiment(self,
                            instance: SATInstance,
                            algorithm: str,
                            instance_name: str,
                            seed: int,
                            **kwargs) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            instance: SAT instance to solve
            algorithm: "GA" or "WOC"
            instance_name: Name of instance
            seed: Random seed
            **kwargs: Algorithm parameters

        Returns:
            ExperimentResult
        """
        random.seed(seed)
        np.random.seed(seed)

        start_time = time.time()

        if algorithm == "GA":
            ga = GeneticAlgorithm(
                instance,
                population_size=kwargs.get('population_size', 50),  # Reduced from 100
                elite_size=kwargs.get('elite_size', 2),
                crossover_rate=kwargs.get('crossover_rate', 0.8),
                mutation_rate=kwargs.get('mutation_rate', 0.01),
                max_generations=kwargs.get('max_generations', 100),  # Reduced from 1000
                stall_generations=kwargs.get('stall_generations', 30),  # Reduced from 50
                local_search_steps=kwargs.get('local_search_steps', 10),  # Reduced from 20
                verbose=kwargs.get('verbose', False)
            )
            best_sol, best_fit, is_sat = ga.run()
            generations = ga.generation
            evaluations = ga.total_evaluations
            convergence_gen = ga.best_generation
            final_diversity = ga.diversity_history[-1] if ga.diversity_history else 0.0

        elif algorithm == "WOC":
            woc = WisdomOfCrowds(
                instance,
                num_experts=kwargs.get('num_experts', 8),  # Reduced from 16
                top_k=kwargs.get('top_k', 3),  # Reduced from 5
                consensus_threshold_high=kwargs.get('threshold_high', 0.7),
                consensus_threshold_low=kwargs.get('threshold_low', 0.3),
                max_iterations=kwargs.get('max_iterations', 2),  # Reduced from 3
                parallel=False,
                verbose=kwargs.get('verbose', False)
            )
            best_sol, best_fit, is_sat = woc.iterative_woc()
            generations = kwargs.get('max_generations', 100) * kwargs.get('num_experts', 8)
            evaluations = 0  # TODO: track evaluations in WoC
            convergence_gen = 0
            final_diversity = 0.0
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        elapsed_time = time.time() - start_time

        # Count satisfied clauses
        evaluator = SATEvaluator(instance, use_weighted=False)
        satisfied_count, _ = evaluator.count_satisfied_clauses(best_sol)

        result = ExperimentResult(
            algorithm=algorithm,
            instance_name=instance_name,
            n_vars=instance.num_vars,
            n_clauses=instance.num_clauses,
            clause_ratio=instance.num_clauses / instance.num_vars,
            success=is_sat,
            best_fitness=best_fit,
            satisfied_clauses=satisfied_count,
            generations=generations,
            evaluations=evaluations,
            time_seconds=elapsed_time,
            convergence_generation=convergence_gen,
            final_diversity=final_diversity
        )

        return result

    def run_comparison_experiment(self,
                                 instance_files: List[str],
                                 num_runs: int = 30,
                                 algorithms: List[str] = ["GA", "WOC"]) -> Dict:
        """
        Run comparison experiments across multiple instances.

        Args:
            instance_files: List of CNF file paths
            num_runs: Number of independent runs per instance
            algorithms: List of algorithms to compare

        Returns:
            Dictionary with aggregated results
        """
        print(f"\nRunning comparison experiments:")
        print(f"  Instances: {len(instance_files)}")
        print(f"  Algorithms: {algorithms}")
        print(f"  Runs per instance: {num_runs}")
        print(f"  Total experiments: {len(instance_files) * len(algorithms) * num_runs}")
        print("=" * 70)

        for instance_file in instance_files:
            instance_name = os.path.basename(instance_file)
            print(f"\nLoading instance: {instance_name}")

            try:
                instance = read_dimacs_cnf(instance_file)
                print(f"  Variables: {instance.num_vars}, Clauses: {instance.num_clauses}")

                for algorithm in algorithms:
                    print(f"\n  Running {algorithm}...")

                    for run_id in range(num_runs):
                        seed = hash((instance_name, algorithm, run_id)) % (2**31)

                        print(f"    Run {run_id + 1}/{num_runs}...", end='', flush=True)
                        result = self.run_single_experiment(
                            instance, algorithm, instance_name, seed
                        )
                        self.results.append(result)
                        print(f" completed in {result.time_seconds:.2f}s (fitness: {result.best_fitness:.4f}, sat: {result.success})")

                        if (run_id + 1) % 5 == 0:
                            avg_time = np.mean([r.time_seconds for r in self.results[-5:]])
                            print(f"    [Progress: {run_id + 1}/{num_runs}, avg time: {avg_time:.2f}s]")
            except Exception as e:
                print(f"  Error with {instance_name}: {e}")
                continue

        # Save results
        self.save_results()

        # Compute statistics
        stats = self.compute_statistics()
        return stats

    def run_ablation_study(self,
                          instance: SATInstance,
                          instance_name: str,
                          num_runs: int = 30) -> Dict:
        """
        Run ablation study to measure component contributions.

        Variants:
        1. Full WoC (baseline)
        2. WoC without skeleton fixing (no consensus-guided restart)
        3. WoC without local search in experts
        4. WoC without diverse experts (single configuration)
        5. WoC with simple voting (no weighted voting)

        Args:
            instance: SAT instance
            instance_name: Name of instance
            num_runs: Number of runs per variant

        Returns:
            Dictionary with ablation results
        """
        print(f"\nRunning ablation study on {instance_name}")
        print(f"  Runs per variant: {num_runs}")
        print("=" * 70)

        variants = {
            "Full_WoC": {"num_experts": 16, "local_search": True},
            "No_Skeleton": {"num_experts": 16, "max_iterations": 1},
            "No_LocalSearch": {"num_experts": 16},  # Disable in GA
            "Single_Expert": {"num_experts": 1},
            "Simple_Voting": {"num_experts": 16}  # Use beta=0
        }

        ablation_results = {}

        for variant_name, params in variants.items():
            print(f"\n  Testing: {variant_name}")
            variant_results = []

            for run_id in range(num_runs):
                seed = hash((instance_name, variant_name, run_id)) % (2**31)

                # TODO: Implement variants with modified WoC
                # For now, run standard WoC
                result = self.run_single_experiment(
                    instance, "WOC", instance_name, seed, **params
                )
                variant_results.append(result)

                if (run_id + 1) % 10 == 0:
                    print(f"    Run {run_id + 1}/{num_runs}")

            ablation_results[variant_name] = variant_results

        # Compute statistics for each variant
        ablation_stats = self.compute_ablation_statistics(ablation_results)

        # Save ablation results
        self.save_ablation_results(ablation_stats, instance_name)

        return ablation_stats

    def run_parameter_sensitivity(self,
                                 instance: SATInstance,
                                 instance_name: str,
                                 parameter: str,
                                 values: List,
                                 num_runs: int = 10) -> Dict:
        """
        Run parameter sensitivity analysis.

        Args:
            instance: SAT instance
            instance_name: Name of instance
            parameter: Parameter name to vary
            values: List of parameter values to test
            num_runs: Runs per parameter value

        Returns:
            Dictionary with sensitivity results
        """
        print(f"\nParameter sensitivity analysis: {parameter}")
        print(f"  Values: {values}")
        print(f"  Runs per value: {num_runs}")
        print("=" * 70)

        sensitivity_results = {}

        for value in values:
            print(f"\n  Testing {parameter}={value}")
            value_results = []

            for run_id in range(num_runs):
                seed = hash((instance_name, parameter, value, run_id)) % (2**31)

                kwargs = {parameter: value}
                result = self.run_single_experiment(
                    instance, "WOC", instance_name, seed, **kwargs
                )
                value_results.append(result)

            sensitivity_results[value] = value_results

            # Print quick stats
            success_rate = np.mean([r.success for r in value_results])
            avg_fitness = np.mean([r.best_fitness for r in value_results])
            print(f"    Success rate: {success_rate:.2%}, Avg fitness: {avg_fitness:.4f}")

        # Save sensitivity results
        self.save_sensitivity_results(sensitivity_results, instance_name, parameter)

        return sensitivity_results

    def compute_statistics(self) -> Dict:
        """Compute aggregate statistics from results."""
        stats = {}

        # Group by algorithm and instance
        by_algorithm = defaultdict(list)
        for result in self.results:
            by_algorithm[result.algorithm].append(result)

        for algo, results in by_algorithm.items():
            stats[algo] = {
                'success_rate': np.mean([r.success for r in results]),
                'avg_fitness': np.mean([r.best_fitness for r in results]),
                'std_fitness': np.std([r.best_fitness for r in results]),
                'avg_satisfied': np.mean([r.satisfied_clauses for r in results]),
                'avg_time': np.mean([r.time_seconds for r in results]),
                'std_time': np.std([r.time_seconds for r in results]),
                'avg_generations': np.mean([r.generations for r in results]),
            }

            # Compute confidence interval
            fitness_vals = [r.best_fitness for r in results]
            if len(fitness_vals) > 1:
                ci = stats.t.interval(0.95, len(fitness_vals)-1,
                                     loc=np.mean(fitness_vals),
                                     scale=stats.sem(fitness_vals))
                stats[algo]['fitness_ci'] = ci

        return stats

    def compute_ablation_statistics(self, ablation_results: Dict) -> Dict:
        """Compute statistics for ablation study."""
        ablation_stats = {}

        for variant, results in ablation_results.items():
            ablation_stats[variant] = {
                'success_rate': np.mean([r.success for r in results]),
                'avg_fitness': np.mean([r.best_fitness for r in results]),
                'std_fitness': np.std([r.best_fitness for r in results]),
                'avg_time': np.mean([r.time_seconds for r in results]),
            }

        return ablation_stats

    def save_results(self):
        """Save results to JSON."""
        results_file = os.path.join(self.output_dir, "experiment_results.json")

        results_dict = [r.to_dict() for r in self.results]

        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    def save_ablation_results(self, stats: Dict, instance_name: str):
        """Save ablation study results."""
        ablation_file = os.path.join(self.output_dir, f"ablation_{instance_name}.json")

        with open(ablation_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Ablation results saved to: {ablation_file}")

    def save_sensitivity_results(self, results: Dict, instance_name: str, parameter: str):
        """Save parameter sensitivity results."""
        sensitivity_file = os.path.join(self.output_dir, f"sensitivity_{parameter}_{instance_name}.json")

        # Convert results to serializable format
        serializable_results = {}
        for value, result_list in results.items():
            serializable_results[str(value)] = [r.to_dict() for r in result_list]

        with open(sensitivity_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Sensitivity results saved to: {sensitivity_file}")


class SATVisualizerGUI:
    """GUI for visualizing SAT solving process."""

    def __init__(self, root: tk.Tk):
        """Initialize GUI."""
        self.root = root
        self.root.title("3-SAT Solver Visualizer")
        self.root.geometry("1400x900")

        self.instance: Optional[SATInstance] = None
        self.solution: Optional[SATSolution] = None
        self.evaluator: Optional[SATEvaluator] = None
        self.ga: Optional[GeneticAlgorithm] = None
        self.woc: Optional[WisdomOfCrowds] = None

        self.setup_ui()

    def setup_ui(self):
        """Setup user interface."""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load CNF", command=self.load_cnf)
        file_menu.add_command(label="Save Results", command=self.save_results_gui)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        run_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Run Standard GA", command=self.run_ga)
        run_menu.add_command(label="Run WoC", command=self.run_woc)
        run_menu.add_command(label="Compare Algorithms", command=self.compare_algorithms)

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Formula and variable view
        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # Formula view
        formula_frame = ttk.LabelFrame(left_panel, text="CNF Formula", padding=5)
        formula_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.formula_text = tk.Text(formula_frame, wrap=tk.WORD, height=20)
        formula_scroll = ttk.Scrollbar(formula_frame, command=self.formula_text.yview)
        self.formula_text.configure(yscrollcommand=formula_scroll.set)
        self.formula_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        formula_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Variable panel
        var_frame = ttk.LabelFrame(left_panel, text="Variables", padding=5)
        var_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.var_text = tk.Text(var_frame, wrap=tk.WORD, height=15)
        var_scroll = ttk.Scrollbar(var_frame, command=self.var_text.yview)
        self.var_text.configure(yscrollcommand=var_scroll.set)
        self.var_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        var_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Right panel: Visualizations
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Notebook for different visualizations
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Evolution curves tab
        self.evolution_frame = ttk.Frame(notebook)
        notebook.add(self.evolution_frame, text="Evolution Curves")

        # Initialize as None, create only when needed
        self.fig_evolution = None
        self.ax_evolution = None
        self.canvas_evolution = None

        # Structure graph tab
        self.structure_frame = ttk.Frame(notebook)
        notebook.add(self.structure_frame, text="Structure Graph")

        # Initialize as None, create only when needed
        self.fig_structure = None
        self.ax_structure = None
        self.canvas_structure = None

        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")

        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD)
        stats_scroll = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Force GUI update
        self.root.update_idletasks()
        print("GUI initialized successfully!")

    def load_cnf(self):
        """Load CNF file."""
        filename = filedialog.askopenfilename(
            title="Select CNF file",
            filetypes=[("CNF files", "*.cnf"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.instance = read_dimacs_cnf(filename)
                self.evaluator = SATEvaluator(self.instance)
                self.status_bar.config(text=f"Loaded: {os.path.basename(filename)}")

                self.display_formula()
                self.draw_structure_graph()

                messagebox.showinfo("Success",
                    f"Loaded instance with {self.instance.num_vars} variables "
                    f"and {self.instance.num_clauses} clauses")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CNF: {e}")

    def display_formula(self):
        """Display CNF formula with highlighting."""
        if not self.instance:
            return

        self.formula_text.delete(1.0, tk.END)
        self.formula_text.tag_config("satisfied", background="lightgreen")
        self.formula_text.tag_config("unsatisfied", background="lightcoral")

        # Display clauses
        for idx, clause in enumerate(self.instance.clauses):
            clause_str = f"C{idx+1}: "

            literals = []
            for lit in clause:
                if lit > 0:
                    literals.append(f"x{lit}")
                else:
                    literals.append(f"¬x{abs(lit)}")

            clause_str += " ∨ ".join(literals) + "\n"

            start_idx = self.formula_text.index(tk.INSERT)
            self.formula_text.insert(tk.END, clause_str)
            end_idx = self.formula_text.index(tk.INSERT)

            # Highlight if solution exists
            if self.solution:
                is_satisfied = self.evaluator.is_clause_satisfied(self.solution, clause)
                tag = "satisfied" if is_satisfied else "unsatisfied"
                self.formula_text.tag_add(tag, start_idx, end_idx)

    def display_variables(self, support: Optional[np.ndarray] = None):
        """Display variables with current assignments and WoC support."""
        if not self.instance:
            return

        self.var_text.delete(1.0, tk.END)

        for i in range(self.instance.num_vars):
            var_str = f"x{i+1}: "

            if self.solution:
                value = self.solution.assignment[i]
                var_str += "True " if value else "False"
            else:
                var_str += "    -   "

            if support is not None:
                # Display support as heatmap bar
                support_val = support[i]
                bar_length = int(support_val * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                var_str += f" [{bar}] {support_val:.2f}"

                if support_val >= 0.7:
                    var_str += " (FIXED True)"
                elif support_val <= 0.3:
                    var_str += " (FIXED False)"
                else:
                    var_str += " (uncertain)"

            var_str += "\n"
            self.var_text.insert(tk.END, var_str)

    def draw_structure_graph(self):
        """Draw bipartite graph of variables and clauses."""
        if not self.instance:
            return

        self.ax_structure.clear()

        # Create bipartite graph
        G = nx.Graph()

        # Add variable nodes
        var_nodes = [f"x{i+1}" for i in range(self.instance.num_vars)]
        clause_nodes = [f"C{i+1}" for i in range(self.instance.num_clauses)]

        G.add_nodes_from(var_nodes, bipartite=0)
        G.add_nodes_from(clause_nodes, bipartite=1)

        # Add edges (variable-clause connections)
        for c_idx, clause in enumerate(self.instance.clauses):
            for literal in clause:
                var_idx = abs(literal) - 1
                G.add_edge(f"x{var_idx+1}", f"C{c_idx+1}")

        # Layout
        pos = {}
        # Variables on left
        for i, node in enumerate(var_nodes[:20]):  # Limit display for large instances
            pos[node] = (0, i)
        # Clauses on right
        for i, node in enumerate(clause_nodes[:20]):
            pos[node] = (1, i)

        # Draw
        nx.draw_networkx_nodes(G, pos, nodelist=var_nodes[:20],
                              node_color='lightblue', node_size=300,
                              ax=self.ax_structure)
        nx.draw_networkx_nodes(G, pos, nodelist=clause_nodes[:20],
                              node_color='lightcoral', node_size=300,
                              ax=self.ax_structure)

        # Draw edges (only for displayed nodes)
        display_edges = [(u, v) for u, v in G.edges()
                        if u in pos and v in pos]
        nx.draw_networkx_edges(G, pos, edgelist=display_edges,
                              alpha=0.3, ax=self.ax_structure)

        # Labels
        labels = {node: node for node in list(pos.keys())}
        nx.draw_networkx_labels(G, pos, labels, font_size=8,
                               ax=self.ax_structure)

        self.ax_structure.set_title("Variable-Clause Structure (First 20 nodes)")
        self.ax_structure.axis('off')
        self.canvas_structure.draw()

    def run_ga(self):
        """Run standard GA."""
        if not self.instance:
            messagebox.showwarning("Warning", "Please load a CNF file first")
            return

        self.status_bar.config(text="Running Standard GA...")
        self.root.update()

        try:
            self.ga = GeneticAlgorithm(
                self.instance,
                population_size=100,
                max_generations=500,
                verbose=True
            )

            self.solution, fitness, is_sat = self.ga.run()

            self.display_formula()
            self.display_variables()
            self.plot_evolution_curves()
            self.display_statistics()

            self.status_bar.config(text=f"GA completed: fitness={fitness:.4f}, sat={is_sat}")

            messagebox.showinfo("GA Complete",
                f"Fitness: {fitness:.4f}\nSatisfying: {is_sat}\n"
                f"Generations: {self.ga.generation}")
        except Exception as e:
            messagebox.showerror("Error", f"GA failed: {e}")
            self.status_bar.config(text="GA failed")

    def run_woc(self):
        """Run WoC."""
        if not self.instance:
            messagebox.showwarning("Warning", "Please load a CNF file first")
            return

        self.status_bar.config(text="Running WoC...")
        self.root.update()

        try:
            self.woc = WisdomOfCrowds(
                self.instance,
                num_experts=8,
                max_iterations=2,
                parallel=False,
                verbose=True
            )

            self.solution, fitness, is_sat = self.woc.iterative_woc()

            self.display_formula()
            self.display_variables(self.woc.variable_support)
            self.display_statistics()

            self.status_bar.config(text=f"WoC completed: fitness={fitness:.4f}, sat={is_sat}")

            messagebox.showinfo("WoC Complete",
                f"Fitness: {fitness:.4f}\nSatisfying: {is_sat}")
        except Exception as e:
            messagebox.showerror("Error", f"WoC failed: {e}")
            self.status_bar.config(text="WoC failed")

    def plot_evolution_curves(self):
        """Plot evolution curves."""
        if not self.ga or not self.ga.fitness_history:
            return

        self.ax_evolution[0].clear()
        self.ax_evolution[1].clear()

        generations = list(range(len(self.ga.fitness_history)))

        # Fitness curve
        self.ax_evolution[0].plot(generations, self.ga.fitness_history,
                                 label='Average Fitness', color='blue')
        self.ax_evolution[0].axhline(y=self.ga.best_fitness,
                                    color='red', linestyle='--',
                                    label=f'Best: {self.ga.best_fitness:.4f}')
        self.ax_evolution[0].set_xlabel('Generation')
        self.ax_evolution[0].set_ylabel('Fitness')
        self.ax_evolution[0].set_title('Fitness Evolution')
        self.ax_evolution[0].legend()
        self.ax_evolution[0].grid(True, alpha=0.3)

        # Diversity curve
        if self.ga.diversity_history:
            self.ax_evolution[1].plot(generations, self.ga.diversity_history,
                                     label='Diversity', color='green')
            self.ax_evolution[1].set_xlabel('Generation')
            self.ax_evolution[1].set_ylabel('Diversity')
            self.ax_evolution[1].set_title('Population Diversity')
            self.ax_evolution[1].legend()
            self.ax_evolution[1].grid(True, alpha=0.3)

        self.fig_evolution.tight_layout()
        self.canvas_evolution.draw()

    def display_statistics(self):
        """Display solution statistics."""
        if not self.solution or not self.evaluator:
            return

        self.stats_text.delete(1.0, tk.END)

        # Basic stats
        fitness, flips, is_sat = self.evaluator.evaluate(self.solution)
        satisfied_count, unsatisfied = self.evaluator.count_satisfied_clauses(self.solution)

        stats_str = "Solution Statistics\n"
        stats_str += "=" * 50 + "\n\n"
        stats_str += f"Variables: {self.instance.num_vars}\n"
        stats_str += f"Clauses: {self.instance.num_clauses}\n"
        stats_str += f"Clause ratio: {self.instance.num_clauses/self.instance.num_vars:.2f}\n\n"

        stats_str += f"Fitness: {fitness:.4f}\n"
        stats_str += f"Satisfied clauses: {satisfied_count}/{self.instance.num_clauses}\n"
        stats_str += f"Unsatisfied clauses: {len(unsatisfied)}\n"
        stats_str += f"Satisfying: {is_sat}\n"
        stats_str += f"Estimated min flips: {flips}\n\n"

        if self.ga:
            stats_str += "GA Statistics\n"
            stats_str += "-" * 50 + "\n"
            stats_str += f"Generations: {self.ga.generation}\n"
            stats_str += f"Evaluations: {self.ga.total_evaluations}\n"
            stats_str += f"Best found at generation: {self.ga.best_generation}\n"
            stats_str += f"Final diversity: {self.ga.diversity_history[-1]:.4f}\n\n"

        if self.woc and self.woc.variable_support is not None:
            stats_str += "WoC Statistics\n"
            stats_str += "-" * 50 + "\n"
            stats_str += f"Number of experts: {self.woc.num_experts}\n"
            stats_str += f"Solutions collected: {len(self.woc.all_solutions)}\n"
            stats_str += f"Variable support mean: {np.mean(self.woc.variable_support):.3f}\n"
            stats_str += f"Variable support std: {np.std(self.woc.variable_support):.3f}\n"

            num_certain = sum(1 for x in self.woc.consensus_assignment if x != -1)
            stats_str += f"Certain variables: {num_certain}/{self.instance.num_vars}\n"

        self.stats_text.insert(tk.END, stats_str)

    def compare_algorithms(self):
        """Run comparison between GA and WoC."""
        if not self.instance:
            messagebox.showwarning("Warning", "Please load a CNF file first")
            return

        # Create progress dialog
        progress = tk.Toplevel(self.root)
        progress.title("Running Comparison")
        progress.geometry("400x150")

        label = ttk.Label(progress, text="Running algorithm comparison...")
        label.pack(pady=20)

        progress_bar = ttk.Progressbar(progress, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()

        self.root.update()

        # Run experiments
        runner = ExperimentRunner()

        results_ga = []
        results_woc = []

        for i in range(5):
            # GA
            result_ga = runner.run_single_experiment(
                self.instance, "GA", "current", i
            )
            results_ga.append(result_ga)

            # WoC
            result_woc = runner.run_single_experiment(
                self.instance, "WOC", "current", i
            )
            results_woc.append(result_woc)

        progress_bar.stop()
        progress.destroy()

        # Display comparison
        comparison_str = "Algorithm Comparison (5 runs each)\n"
        comparison_str += "=" * 50 + "\n\n"

        comparison_str += "Standard GA:\n"
        comparison_str += f"  Success rate: {np.mean([r.success for r in results_ga]):.2%}\n"
        comparison_str += f"  Avg fitness: {np.mean([r.best_fitness for r in results_ga]):.4f}\n"
        comparison_str += f"  Avg time: {np.mean([r.time_seconds for r in results_ga]):.2f}s\n\n"

        comparison_str += "WoC:\n"
        comparison_str += f"  Success rate: {np.mean([r.success for r in results_woc]):.2%}\n"
        comparison_str += f"  Avg fitness: {np.mean([r.best_fitness for r in results_woc]):.4f}\n"
        comparison_str += f"  Avg time: {np.mean([r.time_seconds for r in results_woc]):.2f}s\n"

        messagebox.showinfo("Comparison Results", comparison_str)

    def save_results_gui(self):
        """Save current results."""
        if not self.solution:
            messagebox.showwarning("Warning", "No solution to save")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            with open(filename, 'w') as f:
                f.write(f"Solution: {self.solution.to_string()}\n")
                if self.ga:
                    f.write(f"Generations: {self.ga.generation}\n")
                    f.write(f"Evaluations: {self.ga.total_evaluations}\n")

            messagebox.showinfo("Success", f"Results saved to {filename}")


def main():
    """Main entry point."""
    print("3-SAT Solver - Main Entry Point")
    print("=" * 70)
    print("\nOptions:")
    print("1. Run GUI")
    print("2. Run batch experiments")
    print("3. Run ablation study")
    print("4. Run parameter sensitivity")

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = input("\nSelect mode (1-4): ").strip()

    if mode == "1":
        # Run GUI
        root = tk.Tk()
        app = SATVisualizerGUI(root)
        root.mainloop()

    elif mode == "2":
        # Run batch experiments
        print("\nRunning batch experiments...")

        # Find CNF files
        data_dir = "data"
        cnf_files = []
        for root_dir, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.cnf'):
                    cnf_files.append(os.path.join(root_dir, file))

        if not cnf_files:
            print("No CNF files found in data directory")
            return

        print(f"Found {len(cnf_files)} CNF files")

        runner = ExperimentRunner()
        stats = runner.run_comparison_experiment(
            cnf_files[:5],  # Limit to first 5 for testing
            num_runs=10,
            algorithms=["GA", "WOC"]
        )

        print("\n" + "=" * 70)
        print("Experiment Summary:")
        for algo, algo_stats in stats.items():
            print(f"\n{algo}:")
            for key, value in algo_stats.items():
                print(f"  {key}: {value}")

    elif mode == "3":
        # Run ablation study
        print("\nRunning ablation study...")

        # Load a test instance
        test_file = "data/test/test_3partition_easy.cnf"
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            return

        instance = read_dimacs_cnf(test_file)
        runner = ExperimentRunner()

        ablation_stats = runner.run_ablation_study(
            instance, "test_instance", num_runs=10
        )

        print("\n" + "=" * 70)
        print("Ablation Study Results:")
        for variant, stats in ablation_stats.items():
            print(f"\n{variant}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    elif mode == "4":
        # Run parameter sensitivity
        print("\nRunning parameter sensitivity analysis...")

        # Load a test instance
        test_file = "data/test/test_3partition_easy.cnf"
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            return

        instance = read_dimacs_cnf(test_file)
        runner = ExperimentRunner()

        # Test different parameters
        parameters = {
            'num_experts': [8, 16, 32],
            'threshold_high': [0.6, 0.7, 0.8],
            'top_k': [3, 5, 7]
        }

        for param, values in parameters.items():
            print(f"\nTesting parameter: {param}")
            sensitivity_results = runner.run_parameter_sensitivity(
                instance, "test_instance", param, values, num_runs=5
            )

    else:
        print("Invalid mode selected")


if __name__ == "__main__":
    main()

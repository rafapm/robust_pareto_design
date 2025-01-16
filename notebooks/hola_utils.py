"""
Copyright 2025 Rafael Perez Martinez

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
---------------------------------------------------------------------------------
This module provides functions and workflows for Pareto efficiency determination,
HOLA optimization runs, and benchmark evaluation. It includes methods for:

- Determining Pareto efficiency from a set of cost arrays.
- Scalarizing objective values based on given targets, limits, and priorities.
- Running single and multiple HOLA optimization trials.
- Plotting and analyzing results of the optimization runs.
- Saving results to CSV files and evaluating performance against benchmarks.
"""

import logging
import timeit
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Determine the Pareto efficiency of each point in a set of costs.

    A point is Pareto efficient if no other point has all cost components
    less than or equal to it, with at least one component being strictly less.

    Parameters
    ----------
    costs : np.ndarray
        A 2D NumPy array where each row represents a point in the cost space.

    Returns
    -------
    np.ndarray
        A boolean array indicating whether each point is Pareto efficient.
    """
    num_points = costs.shape[0]
    is_efficient = np.ones(num_points, dtype=bool)

    for i, cost in enumerate(costs):
        if is_efficient[i]:
            # A point is not efficient if any other point dominates it
            is_efficient[i] = np.all(np.any(costs >= cost, axis=1))

    return is_efficient


def make_array_decreasing(arr: np.ndarray) -> np.ndarray:
    """
    Transform the array to be monotonically decreasing by taking the minimum
    accumulated value at each index.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        A monotonically decreasing array.
    """
    return np.minimum.accumulate(arr)


def scalarize(target: float, limit: float, priority: float, values: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Scalarize objective values based on given target, limit, and priority.

    Parameters
    ----------
    target : float
        The target value for the objective.
    limit : float
        The limit value for the objective.
    priority : float
        The priority or weight for the objective.
    values : Union[np.ndarray, List[float]]
        An array or list of objective values to be scalarized.

    Returns
    -------
    np.ndarray
        The scalarized objective values.
    """
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    if limit < target:
        result = np.where(
            values < limit,
            np.inf,
            np.where(values > target, 0.0, priority - priority * (values - limit) / (target - limit)),
        )
    else:
        result = np.where(
            values < target, 0.0, np.where(values > limit, np.inf, priority * (values - target) / (limit - target))
        )
    return result


def calculate_score(objectives: Dict[str, Dict[str, float]], data_arrays: Dict[str, np.ndarray]) -> float:
    """
    Calculate the total score across all objectives.

    Parameters
    ----------
    objectives : dict
        A dictionary of objectives, where keys are objective names and values
        are dictionaries with 'target', 'limit', and 'priority' keys.
    data_arrays : dict
        A dictionary of data arrays corresponding to objectives and parameters.

    Returns
    -------
    float
        The total score across all objectives.
    """
    scores = []
    for obj_name, obj_conf in objectives.items():
        target = obj_conf["target"]
        limit = obj_conf["limit"]
        priority = obj_conf["priority"]

        values = data_arrays[obj_name]
        obj_score = scalarize(target, limit, priority, values)
        scores.append(obj_score)

    total_score = np.sum(scores, axis=0)
    optimal_score = np.min(total_score)
    logging.info("Optimal score: %s", optimal_score)
    return optimal_score


def run_hola(
    num_runs: int,
    objectives: Dict[str, Dict[str, float]],
    params_config: Dict[str, Any],
    data_arrays: Dict[str, np.ndarray],
    tune_function: Callable,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Run the HOLA optimization process.

    Parameters
    ----------
    num_runs : int
        Number of optimization runs (trials).
    objectives : dict
        Dictionary of objectives configuration.
    params_config : dict
        Dictionary specifying parameter configurations.
    data_arrays : dict
        Dictionary of data arrays for all parameters and objectives.
    tune_function : callable
        A tuning function that implements the optimization logic.

    Returns
    -------
    tuple
        A tuple containing:
        - leaderboard (pd.DataFrame): DataFrame of run results.
        - best_params (dict): Dictionary of best found parameters.
        - best_objs (dict): Dictionary of objective scores for best parameters.
    """
    # Ensure all data_arrays are NumPy arrays
    for k, v in data_arrays.items():
        if not isinstance(v, np.ndarray):
            data_arrays[k] = v.to_numpy()

    arr_vds = data_arrays["vds"]
    arr_wg = data_arrays["wg"]
    arr_nfing = data_arrays["nfing"]
    arr_gdg = data_arrays["gdg"]
    arr_gsg = data_arrays["gsg"]

    def look_up(vds: int, wg: int, nfing: int, gdg: int, gsg: int) -> Dict[str, float]:
        """
        Find corresponding objective values for given parameters.

        Parameters
        ----------
        vds : int
        wg : int
        nfing : int
        gdg : int
        gsg : int

        Returns
        -------
        dict
            Dictionary of objective values.
        """
        indices = np.where(
            (arr_vds == vds) & (arr_wg == wg) & (arr_nfing == nfing) & (arr_gdg == gdg) & (arr_gsg == gsg)
        )[0]

        if indices.size == 0:
            logging.info(
                "Device design does not meet imposed constraints: vds=%s, wg=%s, nfing=%s, gdg=%s, gsg=%s",
                vds,
                wg,
                nfing,
                gdg,
                gsg,
            )
            raise ValueError("No valid combination found.")

        index = indices[0]

        obj_values = {obj_name: data_arrays[obj_name][index] for obj_name in objectives.keys()}
        return obj_values

    def evaluate_objectives(vds: int, wg: int, nfing: int, gdg: int, gsg: int) -> Dict[str, float]:
        """
        Evaluate the objectives based on input parameters.

        Parameters
        ----------
        vds : int
        wg : int
        nfing : int
        gdg : int
        gsg : int

        Returns
        -------
        dict
            Dictionary of objective values.
        """
        try:
            return look_up(vds, wg, nfing, gdg, gsg)
        except ValueError:
            # If no combo is found, return zeros for all objectives
            return {obj_name: 0.0 for obj_name in objectives.keys()}

    # Start timing
    start_time = timeit.default_timer()

    # Perform tuning
    tuner = tune_function(evaluate_objectives, params_config, objectives, num_runs=num_runs, n_jobs=1)

    # End timing
    elapsed_time = timeit.default_timer() - start_time
    logging.info("Time taken for tuning: %.2f seconds", elapsed_time)

    # Display best parameters and scores
    logging.info("Best Parameters:")
    logging.info(tuner.get_best_params())

    logging.info("Best Objectives:")
    logging.info(tuner.get_best_scores())

    # Retrieve and return leaderboard
    leaderboard = tuner.get_leaderboard()
    return leaderboard, tuner.get_best_params(), tuner.get_best_scores()


def plot_leaderboard(leaderboard: pd.DataFrame, y_val: float, trials: int, plot_enable: bool) -> None:
    """
    Plot the leaderboard scores and print specific trial scores.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        DataFrame containing the leaderboard scores.
    y_val : float
        The optimal score to compare against.
    trials : int
        Number of trials.
    plot_enable : bool
        If True, a plot is displayed.
    """
    score_tally = leaderboard.sort_values("run", ascending=True)["score"].values
    score_tally_decreasing = make_array_decreasing(np.array(score_tally))

    if plot_enable:
        trial_range = np.arange(1, trials + 1)
        _, ax = plt.subplots(figsize=(16, 10))
        plt.plot(trial_range, score_tally_decreasing, label=r"Score$^\star$", color="blue", linewidth=4.5)
        plt.axhline(y=y_val, linestyle="--", label="Optimal Score", color="red", linewidth=4.5)
        plt.xlabel("Number of Trials (n)")
        plt.ylabel(r"Score$^\star$")

        ax.set_xlim([1, trials])
        ax.set_ylim([0, 0.7])

        x_step = 25
        x_ticks = np.arange(0, trials + x_step, x_step)
        ax.set_xticks(x_ticks)

        ax.set_xlabel("Number of Trials (n)", fontsize=36, color="black")
        ax.set_ylabel(r"Score$^\star$", fontsize=36, color="black")

        plt.tick_params(
            axis="both", which="major", direction="in", length=6, width=4.75, colors="black", labelsize=32, pad=12
        )

        spine_width = 4.75
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1, colors="black")

        ax.grid(True, which="both", color="grey", linestyle="--", linewidth=2.25, zorder=0, alpha=0.4)

        ax.legend(loc="upper right", fontsize=30, framealpha=1, facecolor="white", edgecolor="white")

        plt.grid(True)
        plt.show()

    # Print specific trial scores depending on the number of trials
    if trials == 50:
        indices_to_print = [0, 9, 19, 29, 39, 49]
    elif trials == 75:
        indices_to_print = [0, 14, 29, 44, 59, 74]
    elif trials == 150:
        indices_to_print = [0, 24, 49, 74, 99, 124, 149]
    else:
        step = max(1, trials // 5)
        indices_to_print = list(range(0, trials, step))

    for i in indices_to_print:
        if i < len(score_tally_decreasing):
            logging.info("The score for trial %s is: %s", i + 1, score_tally_decreasing[i])


def run_hola_trials(
    trials: int,
    objectives: Dict[str, Dict[str, float]],
    tune_function: Callable,
    data_arrays: Dict[str, np.ndarray],
    params_config: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Run multiple HOLA optimization trials.

    Parameters
    ----------
    trials : int
        Number of trials.
    objectives : dict
        Objectives configuration dictionary.
    tune_function : callable
        Tuning function to perform optimization.
    data_arrays : dict
        Dictionary of data arrays.
    params_config : dict
        Parameter configuration dictionary.

    Returns
    -------
    tuple
        (score_tally_decreasing, best_params, best_objs)
    """
    leaderboard, best_params, best_objs = run_hola(
        num_runs=trials,
        objectives=objectives,
        params_config=params_config,
        data_arrays=data_arrays,
        tune_function=tune_function,
    )

    optimal_score = calculate_score(objectives, data_arrays)
    plot_leaderboard(leaderboard=leaderboard, y_val=optimal_score, trials=trials, plot_enable=False)

    score_tally = leaderboard.sort_values("run", ascending=True)["score"].values
    score_tally_decreasing = make_array_decreasing(np.array(score_tally))
    return score_tally_decreasing, best_params, best_objs


def run_multiple_hola_trials(
    n_runs: int,
    trials: int,
    objectives: Dict[str, Dict[str, float]],
    tune_function: Callable,
    data_arrays: Dict[str, np.ndarray],
    params_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run multiple independent HOLA optimization trials and collect statistics.

    Parameters
    ----------
    n_runs : int
        Number of runs.
    trials : int
        Number of trials per run.
    objectives : dict
        Objectives configuration.
    tune_function : callable
        Tuning function.
    data_arrays : dict
        Data arrays.
    params_config : dict
        Parameter configuration.

    Returns
    -------
    dict
        A dictionary containing arrays of all score tallies, mean and std
        scores across runs, final scores, and best parameters and objectives.
    """
    all_score_tallies = []
    final_scores = []
    all_best_params = []
    all_best_objs = []

    for run in range(1, n_runs + 1):
        logging.info("Starting run %s/%s", run, n_runs)
        score_tally, best_params, best_objs = run_hola_trials(
            trials=trials,
            objectives=objectives,
            tune_function=tune_function,
            data_arrays=data_arrays,
            params_config=params_config,
        )
        all_score_tallies.append(score_tally)
        final_scores.append(score_tally[-1])
        all_best_params.append(best_params)
        all_best_objs.append(best_objs)
        logging.info("Completed run %s/%s", run, n_runs)

    all_score_tallies = np.array(all_score_tallies)
    mean_scores = np.mean(all_score_tallies, axis=0)
    std_scores = np.std(all_score_tallies, axis=0)

    final_mean = np.mean(final_scores)
    final_std = np.std(final_scores)

    logging.info("Final Scores across %s runs: Mean = %.2f, Std = %.2f", n_runs, final_mean, final_std)
    return {
        "all_score_tallies": all_score_tallies,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "final_scores": final_scores,
        "final_mean": final_mean,
        "final_std": final_std,
        "all_best_params": all_best_params,
        "all_best_objs": all_best_objs,
    }


def save_results_to_csv(results: Dict[str, Any], output_dir: str = ".") -> None:
    """
    Save the results dictionary to CSV files.

    Parameters
    ----------
    results : dict
        Results dictionary from `run_multiple_hola_trials`.
    output_dir : str, optional
        Directory to save CSV files, by default "."
    """
    output_path = Path(output_dir)

    # Create the directory if it doesn't exist
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {output_path.resolve()}")
        except Exception as e:
            logging.error("Failed to create directory %s: %s", output_path, e)
            raise

    all_score_tallies = results["all_score_tallies"]
    mean_scores = results["mean_scores"]
    std_scores = results["std_scores"]
    final_scores = results["final_scores"]
    final_mean = results["final_mean"]
    final_std = results["final_std"]
    all_best_params = results["all_best_params"]
    all_best_objs = results["all_best_objs"]

    # All Score Tallies
    df_all_scores = pd.DataFrame(
        all_score_tallies,
        columns=[f"Trial_{i + 1}" for i in range(all_score_tallies.shape[1])],
        index=[f"Run_{i + 1}" for i in range(all_score_tallies.shape[0])],
    )
    df_all_scores.index.name = "Run"
    all_score_tallies_path = output_path / "all_score_tallies.csv"
    df_all_scores.to_csv(all_score_tallies_path)
    print(f"Saved all_score_tallies.csv to {all_score_tallies_path}")

    # Mean and Std Scores
    trials = mean_scores.shape[0]
    df_mean_std = pd.DataFrame(
        {"Trial": [f"Trial_{i + 1}" for i in range(trials)], "Mean_Score": mean_scores, "Std_Score": std_scores}
    )
    mean_std_scores_path = output_path / "mean_std_scores.csv"
    df_mean_std.to_csv(mean_std_scores_path, index=False)
    print(f"Saved mean_std_scores.csv to {mean_std_scores_path}")

    # Final Scores
    n_runs = len(final_scores)
    df_final_scores = pd.DataFrame(
        {
            "Run": [f"Run_{i + 1}" for i in range(n_runs)],
            "Final_Score": final_scores,
            "Best_Params": all_best_params,
            "Best_Objs": all_best_objs,
        }
    )
    final_scores_path = output_path / "final_scores.csv"
    df_final_scores.to_csv(final_scores_path, index=False)
    print(f"Saved final_scores.csv to {final_scores_path}")

    # Final Statistics
    df_final_stats = pd.DataFrame({"Statistic": ["Final_Mean", "Final_Std"], "Value": [final_mean, final_std]})
    final_stats_path = output_path / "final_statistics.csv"
    df_final_stats.to_csv(final_stats_path, index=False)
    print(f"Saved final_statistics.csv to {final_stats_path}")


def plot_benchmark_results(
    mean_scores: np.ndarray,
    std_scores: np.ndarray,
    trials: int,
    objectives: Dict[str, Dict[str, float]],
    data_arrays: Dict[str, np.ndarray],
) -> None:
    """
    Plot the benchmark results showing mean and standard deviation across runs.

    Parameters
    ----------
    mean_scores : np.ndarray
        Array of mean scores across runs.
    std_scores : np.ndarray
        Array of standard deviations of the scores across runs.
    trials : int
        Number of trials.
    objectives : dict
        Objectives configuration.
    data_arrays : dict
        Data arrays.
    """
    trial_range = np.arange(1, trials + 1)
    _, ax = plt.subplots(figsize=(16, 10))

    ax.plot(trial_range, mean_scores, label=r"Mean Score$^\star$", color="blue", linewidth=4.5)

    optimal_score = calculate_score(objectives, data_arrays)
    plt.axhline(y=optimal_score, color="r", linestyle="--", linewidth=4.5, label="Optimal Score", zorder=3)

    ax.fill_between(
        trial_range, mean_scores - std_scores, mean_scores + std_scores, color="blue", alpha=0.15, label="Std. Dev."
    )

    # Add boundary lines for standard deviation
    ax.plot(
        trial_range,
        mean_scores - std_scores,
        color="blue",
        linewidth=2.0,
        linestyle="-",
        alpha=0.3,
    )
    ax.plot(
        trial_range,
        mean_scores + std_scores,
        color="blue",
        linewidth=2.0,
        linestyle="-",
        alpha=0.3,
    )

    ax.set_xlabel("Number of Trials (n)", fontsize=36, color="black")
    ax.set_ylabel(r"Score$^\star$", fontsize=36, color="black")

    ax.set_xlim([1, trials])
    ax.set_ylim([0, 1])

    if trials == 50:
        x_step = 10
    if trials == 75:
        x_step = 15
    elif trials > 100:
        x_step = 25
    else:
        x_step = 10
    x_ticks = np.arange(0, trials + x_step, x_step)
    ax.set_xticks(x_ticks)

    plt.tick_params(
        axis="both", which="major", direction="in", length=6, width=4.75, colors="black", labelsize=32, pad=12
    )

    spine_width = 4.75
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1, colors="black")

    ax.grid(True, which="both", color="grey", linestyle="--", linewidth=2.25, zorder=0, alpha=0.4)

    ax.legend(loc="upper right", fontsize=30, framealpha=1, facecolor="white", edgecolor="white")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_threshold(optimal_score: float, tolerance: float = 0.025) -> float:
    """
    Compute a threshold for being within a certain tolerance of the optimal score.

    Parameters
    ----------
    optimal_score : float
        The optimal score from calculations.
    tolerance : float, optional
        Tolerance fraction, by default 0.025.

    Returns
    -------
    float
        The threshold for being considered "close enough" to the optimal score.
    """
    if optimal_score == 0.0:
        return tolerance
    else:
        return optimal_score * (1 + tolerance)


def evaluate_runs(
    objectives: Dict[str, Dict[str, float]],
    data_arrays: Dict[str, np.ndarray],
    benchmark_results: Dict[str, Any],
    check_points: List[int],
    tolerance: Optional[float] = None,
    compute_threshold_func: Optional[Callable[[float, float], float]] = None,
    score_calculation_func: Optional[Callable[[Dict[str, Dict[str, float]], Dict[str, np.ndarray]], float]] = None,
) -> Tuple[float, Optional[float]]:
    """
    Evaluate benchmark runs against the optimal score and an optional tolerance threshold.

    Parameters
    ----------
    objectives : dict
        Objectives configuration.
    data_arrays : dict
        Data arrays.
    benchmark_results : dict
        Results from `run_multiple_hola_trials`.
    check_points : list
        List of trial numbers to evaluate.
    tolerance : float, optional
        Tolerance fraction, by default None.
    compute_threshold_func : callable, optional
        Function to compute threshold, by default `compute_threshold`.
    score_calculation_func : callable, optional
        Function to calculate the optimal score, must be provided.

    Returns
    -------
    tuple
        (optimal_score, threshold)
    """
    if compute_threshold_func is None:
        compute_threshold_func = compute_threshold

    if score_calculation_func is None:
        raise ValueError("score_calculation_func must be provided.")

    optimal_score = score_calculation_func(objectives, data_arrays)

    threshold = None
    if tolerance is not None:
        threshold = compute_threshold_func(optimal_score, tolerance)

    all_score_tallies = benchmark_results["all_score_tallies"]
    total_runs = all_score_tallies.shape[0]

    for t in check_points:
        trial_index = t - 1  # zero-based indexing
        if threshold is not None:
            num_runs_close = np.sum(all_score_tallies[:, trial_index] <= threshold)
            print(
                f"After {t} trials, {num_runs_close} out of {total_runs} runs are within "
                f"{100 * tolerance}% of the optimal score (optimal_score={optimal_score:.3f})."
            )

        else:
            num_runs_reached = np.sum(all_score_tallies[:, trial_index] <= optimal_score)
            print(f"After {t} trials, {num_runs_reached} out of {total_runs} runs reached the optimal score.")

    return optimal_score, threshold


def evaluate_mean_scores(
    benchmark_results: Dict[str, Any],
    check_points: List[int]
) -> None:
    """
    Print the mean scores at certain trial checkpoints.

    Parameters
    ----------
    benchmark_results : dict
        Results from `run_multiple_hola_trials`.
        Expected to contain a list/array "mean_scores" where mean_scores[0] is trial #1.
    check_points : list of int
        List of 1-based trial numbers to evaluate.

    Returns
    -------
    None
    """
    mean_scores = benchmark_results["mean_scores"]

    for t in check_points:
        # Here, mean_scores[ t - 1 ] corresponds to trial #t
        print(f"After {t} trials, the Mean Score: {mean_scores[t - 1]:.3f}")


def subplot_data(
    dp: np.ndarray,
    pp: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    x_ticks: List[Optional[np.ndarray]],
    y_ticks: List[Optional[np.ndarray]],
    x_ranges: List[Tuple[int, int]],
    y_ranges: List[Tuple[int, int]],
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Plots scatter plots on a 2x2 grid of subplots using data from dp and pp arrays.

    Parameters:
    - dp (np.ndarray): Dominated points data array with at least 3 columns.
    - pp (np.ndarray): Pareto points data array with at least 3 columns.
    - x_labels (List[str]): Labels for the x-axes of the subplots.
    - y_labels (List[str]): Labels for the y-axes of the subplots.
    - x_ticks (List[Optional[np.ndarray]]): Tick positions for the x-axes. Use None or empty array for no ticks.
    - y_ticks (List[Optional[np.ndarray]]): Tick positions for the y-axes. Use None or empty array for no ticks.
    - x_ranges (List[Tuple[int, int]]): Axis limits for the x-axes.
    - y_ranges (List[Tuple[int, int]]): Axis limits for the y-axes.
    - figsize (Tuple[int, int], optional): Size of the figure. Defaults to (16, 10).
    - save_path (Optional[str], optional): Path to save the figure. If None, the figure is not saved.

    Returns:
    - None
    """
    num_subplots = 4  # 2x2 grid
    if not (
        len(x_labels) == len(y_labels) == len(x_ticks) == len(y_ticks) == len(x_ranges) == len(y_ranges) == num_subplots
    ):
        raise ValueError(f"All input lists must have exactly {num_subplots} elements.")

    # Extract data for plotting
    data_x = [dp[:, 0], dp[:, 0], dp[:, 0], dp[:, 1]]  # data1_x  # data2_x  # data3_x  # data4_x

    data_y = [dp[:, 1], dp[:, 1], dp[:, 2], dp[:, 2]]  # data1_y  # data2_y  # data3_y  # data4_y

    data_x2 = [pp[:, 0], pp[:, 0], pp[:, 0], pp[:, 1]]  # data1_x2  # data2_x2  # data3_x2  # data4_x2

    data_y2 = [pp[:, 1], pp[:, 1], pp[:, 2], pp[:, 2]]  # data1_y2  # data2_y2  # data3_y2  # data4_y2

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    axes = axs.flatten()

    subplot_labels = ["(a)", "", "(b)", "(c)"]
    x_padding = 0.02  # Horizontal padding
    y_padding = -0.04  # Vertical padding

    for i in range(num_subplots):
        ax = axes[i]

        if i == 1:
            # Remove the second subplot (top-right)
            fig.delaxes(ax)
            continue

        # Scatter plot for dominated points
        ax.scatter(
            data_x[i], data_y[i], color="blue", label="Dominated", edgecolors="black", linewidths=2.5, s=325, zorder=2
        )

        # Scatter plot for Pareto points
        ax.scatter(
            data_x2[i], data_y2[i], color="red", label="Pareto", edgecolors="black", linewidths=2.5, s=325, zorder=3
        )

        # Set ticks if provided
        if x_ticks[i] is not None and x_ticks[i].size > 0:
            ax.set_xticks(x_ticks[i])
        else:
            ax.set_xticks([])

        if y_ticks[i] is not None and y_ticks[i].size > 0:
            ax.set_yticks(y_ticks[i])
        else:
            ax.set_yticks([])

        # Set axis limits
        ax.set_xlim(x_ranges[i])
        ax.set_ylim(y_ranges[i])

        # Set labels
        if x_labels[i]:
            ax.set_xlabel(x_labels[i], fontsize=36, color="black")
        if y_labels[i]:
            ax.set_ylabel(y_labels[i], fontsize=36, color="black")

        # Customize spines
        line_width = 4.75
        for spine in ax.spines.values():
            spine.set_linewidth(line_width)

        # Customize tick parameters
        ax.tick_params(
            axis="both", which="major", direction="in", length=6, width=4.75, colors="black", labelsize=32, pad=15
        )
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1, colors="black")

        # Add grid
        ax.grid(True, which="both", color="grey", linestyle="--", linewidth=2.25, zorder=0, alpha=0.4)

        # Annotate subplot label
        ax.annotate(
            subplot_labels[i],
            xy=(0 + x_padding, 1 + y_padding),
            xycoords="axes fraction",
            fontsize=34,
            color="black",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),  # Background for visibility
        )

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Save the figure if a save path is provided
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")


def plot_pareto_frontier_3d(
    data: np.ndarray,
    pareto_front: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str,
    x_ticks: np.ndarray,
    y_ticks: np.ndarray,
    z_ticks: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    figsize: Tuple[int, int] = (7, 7),
    dpi: int = 1000,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a 3D scatter plot of data points and Pareto frontier points.

    Parameters:
    - data (np.ndarray): Array of data points with shape (n_samples, 3).
    - pareto_front (np.ndarray): Array of Pareto frontier points with shape (m_samples, 3).
    - x_label (str): Label for the X-axis.
    - y_label (str): Label for the Y-axis.
    - z_label (str): Label for the Z-axis.
    - x_ticks (np.ndarray): Tick positions for the X-axis.
    - y_ticks (np.ndarray): Tick positions for the Y-axis.
    - z_ticks (np.ndarray): Tick positions for the Z-axis.
    - x_range (Tuple[float, float]): Axis limits for the X-axis.
    - y_range (Tuple[float, float]): Axis limits for the Y-axis.
    - z_range (Tuple[float, float]): Axis limits for the Z-axis.
    - figsize (Tuple[int, int], optional): Size of the figure in inches. Defaults to (7, 7).
    - dpi (int, optional): Dots per inch for the figure. Defaults to 1000.
    - save_path (Optional[str], optional): Path to save the figure. If None, the figure is not saved.

    Returns:
    - None
    """

    # Create a new figure with 3D projection
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot for all data points
    ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        c="blue",
        s=250,
        edgecolors="black",
        linewidths=1.5,
        zorder=2,
        alpha=0.1,
        label="Data Points",
    )

    # Scatter plot for Pareto frontier points
    ax.scatter(
        pareto_front[:, 0],
        pareto_front[:, 1],
        pareto_front[:, 2],
        c="red",
        s=250,
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
        alpha=1,
        label="Pareto Frontier",
    )

    # Set axis labels with specified font size, color, and padding
    ax.set_xlabel(x_label, fontsize=24, color="black", labelpad=20)
    ax.set_ylabel(y_label, fontsize=24, color="black", labelpad=20)
    ax.set_zlabel(z_label, fontsize=24, color="black", labelpad=20)

    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)

    # Set axis ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    # Rotate x and y tick labels for better readability
    for t in ax.get_xticklabels():
        t.set_rotation(45)
    for t in ax.get_yticklabels():
        t.set_rotation(45)

    # Customize tick parameters
    ax.tick_params(axis="both", which="major", direction="in", length=0, width=0, colors="black", labelsize=18)
    ax.tick_params(axis="x", which="major", pad=-2)
    ax.tick_params(axis="z", which="major", pad=7)
    ax.tick_params(axis="both", which="minor", direction="in", length=0, width=0, colors="black")

    # Customize pane colors to white and ensure visibility of ticks and panes
    pane_color = (1.0, 1.0, 1.0, 1.0)  # RGBA for white
    ax.w_xaxis.set_pane_color(pane_color)
    ax.w_yaxis.set_pane_color(pane_color)
    ax.w_zaxis.set_pane_color(pane_color)

    for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        for t in a.get_ticklines() + a.get_ticklabels():
            t.set_visible(True)
        a.line.set_visible(True)
        a.pane.set_visible(True)

    # Add grid with specified style
    ax.grid(True, linestyle="--", color="gray", linewidth=4.25, alpha=0.1)
    ax.xaxis._axinfo["grid"]["linestyle"] = "--"
    ax.yaxis._axinfo["grid"]["linestyle"] = "--"
    ax.zaxis._axinfo["grid"]["linestyle"] = "--"

    plt.tight_layout()
    plt.show()

    # Save the figure if a save path is provided
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")


def calculate_y_range_and_ticks(
    y_data: List[float], std_data: List[float], max_ticks: int = 6, scale: float = 0.25
) -> Tuple[Tuple[float, float], np.ndarray]:
    """
    Calculate y-axis range and ticks based on mean and standard deviation data.

    Parameters:
    - y_data (List[float]): Mean scores.
    - std_data (List[float]): Standard deviation of scores.
    - max_ticks (int, optional): Maximum number of ticks. Defaults to 6.
    - scale (float, optional): Scale factor for calculating tick steps. Defaults to 0.25.

    Returns:
    - Tuple containing:
        - y_range (Tuple[float, float]): (y_min, y_max) for the y-axis.
        - y_ticks (np.ndarray): Array of tick positions.
    """
    y_data_np = np.array(y_data)
    std_data_np = np.array(std_data)

    # Filter out invalid values (NaN or Inf)
    valid_mask = np.isfinite(y_data_np) & np.isfinite(std_data_np)
    if not np.any(valid_mask):
        raise ValueError("No valid data points in y_data or std_data.")

    y_data_filtered = y_data_np[valid_mask]
    std_data_filtered = std_data_np[valid_mask]

    # Calculate y_min and y_max with margin
    y_min = np.min(y_data_filtered - std_data_filtered)
    y_max = np.max(y_data_filtered + std_data_filtered)
    margin = (y_max - y_min) * 0.075

    if margin == 0:
        margin = 0.05  # Default margin if data has no variation

    y_min -= margin
    y_max += margin

    # Apply scaling
    y_min = np.floor(y_min / scale) * scale
    y_min = max(y_min, 0)  # Ensure y_min is not negative
    y_max = np.ceil(y_max / scale) * scale

    # Calculate step size for ticks
    range_span = y_max - y_min
    step = scale
    while range_span / step > max_ticks:
        step *= 2

    y_ticks = np.arange(y_min, y_max + step, step)

    return (y_min, y_max), y_ticks


def subplot_benchmark_results(
    trials: int,
    benchmark_results: List[Dict[str, List[float]]],
    optimal_scores: List[float],
    x_labels: List[str],
    y_labels: List[str],
    x_ticks: List[np.ndarray],
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Plots benchmark results across multiple trials on a 2x2 grid of subplots.

    Parameters:
    - trials (int): Total number of trials.
    - benchmark_results (List[Dict[str, List[float]]]):
        A list containing four dictionaries, each with keys "mean_scores" and "std_scores".
    - optimal_scores (List[float]): A list of four optimal scores corresponding to each subplot.
    - x_labels (List[str]): Labels for the x-axes of the subplots.
    - y_labels (List[str]): Labels for the y-axes of the subplots.
    - x_ticks (List[np.ndarray]): Tick positions for the x-axes.
    - figsize (Tuple[int, int], optional): Size of the figure in inches. Defaults to (16, 10).
    - save_path (Optional[str], optional): Path to save the figure. If None, the figure is not saved.

    Returns:
    - None
    """
    if not len(benchmark_results) == len(optimal_scores) == len(x_labels) == len(y_labels) == len(x_ticks) == 4:
        raise ValueError(
            "benchmark_results, optimal_scores, x_labels, y_labels, and x_ticks must all have exactly four elements."
        )

    n_trials = range(1, trials + 1)
    xx = [0, trials]

    # Extract data for plotting
    data_x = [list(n_trials) for _ in range(4)]
    data_y = [br["mean_scores"] for br in benchmark_results]
    data_std = [br["std_scores"] for br in benchmark_results]
    yy = optimal_scores

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs_flat = axs.flatten()

    # Labels for annotations
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    x_padding = 0.02  # Horizontal padding
    y_padding = -0.04  # Vertical padding

    for i, ax in enumerate(axs_flat):
        # Get mean and std data
        mean_scores = data_y[i]
        std_scores = data_std[i]
        trial_range = data_x[i]

        # Calculate y-axis range and ticks
        y_range, y_ticks = calculate_y_range_and_ticks(mean_scores, std_scores, max_ticks=5, scale=0.15)

        # Plot mean scores
        ax.plot(trial_range, mean_scores, color="blue", linewidth=6.5, zorder=2, label=r"Mean Score$^\star$")

        # Add standard deviation shading
        ax.fill_between(
            trial_range,
            np.array(mean_scores) - np.array(std_scores),
            np.array(mean_scores) + np.array(std_scores),
            color="blue",
            alpha=0.15,
            label="Std. Dev.",
        )

        # Add boundary lines for standard deviation
        ax.plot(
            trial_range,
            np.array(mean_scores) - np.array(std_scores),
            color="blue",
            linewidth=4.0,
            linestyle="-",
            alpha=0.3,
        )
        ax.plot(
            trial_range,
            np.array(mean_scores) + np.array(std_scores),
            color="blue",
            linewidth=4.0,
            linestyle="-",
            alpha=0.3,
        )

        # Plot optimal score
        ax.plot(xx, [yy[i]] * 2, "r--", linewidth=4.5, zorder=3, label="Optimal Score")

        # Set ticks and limits
        ax.set_xticks(x_ticks[i])
        ax.set_yticks(y_ticks)
        ax.set_xlim((0, trials))
        ax.set_ylim(y_range)

        # Set labels
        ax.set_xlabel(x_labels[i], fontsize=36, color="black")
        ax.set_ylabel(y_labels[i], fontsize=36, color="black")

        # Customize spines
        line_width = 4.75
        for spine in ax.spines.values():
            spine.set_linewidth(line_width)

        # Customize tick parameters
        ax.tick_params(
            axis="both", which="major", direction="in", length=6, width=4.75, colors="black", labelsize=32, pad=12
        )
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1, colors="black")

        # Add legend
        ax.legend(loc="upper right", fontsize=23, framealpha=1, facecolor="white", edgecolor="white")

        # Add grid
        ax.grid(True, which="both", color="grey", linestyle="--", linewidth=2.25, zorder=0, alpha=0.4)

        # Annotate subplot label
        ax.annotate(
            subplot_labels[i],
            xy=(0 + x_padding, 1 + y_padding),
            xycoords="axes fraction",
            fontsize=34,
            color="black",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),  # Background for visibility
        )

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Save the figure if a save path is provided
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")

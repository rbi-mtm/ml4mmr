""" Helper functions for the tutorial on coarse-grained molecular optimization """
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from subprocess import run
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import logging
logging.getLogger("pymbar").setLevel(logging.ERROR)
from alchemlyb.estimators import MBAR
from alchemlyb.parsing.gmx import extract_u_nk

def run_molecule_simulations(molecule: str, delete_on_failure: bool = False):
    """
    Run all for thermodynamic integration required simulations for a two-bead molecule
    in water, hexane, and a mixture. Since minimized structure files are provided, the
    energy minimization step is skipped.
    :param molecule: A string representing the two bead molecule in the format 'A-B'.
    :param delete_on_failure: Delete the system simulation directory if simulation fails.
    """
    try:
        molecule_path = Path('simulations') / molecule
        for system in ['water', 'hexane', 'mixture']:
            """ Directory setup """
            system_path = molecule_path / system
            if system_path.exists():
                continue
            system_path.mkdir(parents=True, exist_ok=True)
            """ Setup the simulation system """
            topology = Path(f'{system}-lig.top').read_text()
            for placeholder, replacement in zip(['B1', 'B2'], molecule.split('-')):
                topology = topology.replace(placeholder, replacement)
            Path(system_path / 'system.top').write_text(topology)
            """ Perform system equilibration """
            command = (f'gmx grompp -f equilibration.mdp -c {system}-lig.gro ' +
                f'-p {system_path}/system.top -o {system_path}/equilibration.tpr ' +
                f'-po {system_path}/equilibration.out.mdp >> {system_path}/simulation.run.log 2>&1 ' +
                f'&& gmx mdrun -deffnm {system_path}/equilibration  -ntmpi 1 ' +
                f'>> {system_path}/simulation.run.log 2>&1')
            print(f'Running simulation 1/9 in {system}', end='\r')
            run(command, shell=True, check=True)
            """ Adjust the number of integration steps depending on the environment. """
            command = 'sed -i "s/^nsteps.*/nsteps                   = {nsteps}/" lambda-run.mdp'
            if system == 'mixture':
                run(command.format(nsteps=30000), shell=True, check=True)
            else:
                run(command.format(nsteps=20000), shell=True, check=True)
            """ Run eight lambda-step simulations """
            for i in range(8):
                lambda_path = system_path / f'lambda{i}'
                lambda_path.mkdir(exist_ok=True)
                command = f'sed -i "s/^init-lambda-state.*/init-lambda-state        = {i}/" lambda-run.mdp'
                run(command, shell=True, check=True)
                command = (f'gmx grompp -f lambda-run.mdp -c {system_path}/equilibration.gro ' +
                    f'-p {system_path}/system.top -o {lambda_path}/production.tpr ' +
                    f'-po {lambda_path}/production.out.mdp >> {lambda_path}/simulation.run.log 2>&1 ' +
                    f'&& gmx mdrun -deffnm {lambda_path}/production -ntmpi 1 ' +
                    f'>> {lambda_path}/simulation.run.log 2>&1')
                print(f'Running simulation {i+2}/9 in {system}', end='\r')
                run(command, shell=True, check=True)
    except Exception as e:
        if delete_on_failure:
            run(f'rm -r {system_path}', shell=True, check=True)
        print(f'Failed to simulate {molecule}, please retry')
        raise e

def calculate_free_energy(molecule: str, system: str,
                          print_result: bool = False) -> tuple[float, float]:
    """
    Calculate solvation free energies for a given molecule and system using the
    MBAR algorithm (https://doi.org/10.1063/1.2978177). This function assumes
    completed simulations.
    :param molecule: A string representing the two bead molecule in the format 'A-B'.
    :param system: One of the three systems: 'water', 'hexane', 'mixture'
    :param print_result: Print free energy and error estimation
    :returns: The solvation free energy and an uncertainty estimate in kcal/mol.
    """
    """ Collect data from simulation output files """
    path = Path('simulations') / molecule / system
    if not path.exists():
        raise ValueError(f"Simulations for {molecule}/{system} not found")
    xvg_files = [p / 'production.xvg' for p in path.iterdir() if p.is_dir()]
    u_nk_list = [extract_u_nk(f, T=300) for f in xvg_files]
    u_nk_combined = pd.concat(u_nk_list)
    """ Perform MBAR calculation """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mbar = MBAR().fit(u_nk_combined)
    """ Extract result and convert unit """
    free_energy = float(mbar.delta_f_.iloc[0, -1]) * 0.5924  # Convert to kcal/mol
    d_free_energy = float(mbar.d_delta_f_.iloc[0, -1]) * 0.5924 # Convert to kcal/mol
    if print_result:
        print(f'dG_{system} = {free_energy:.3f} ± {d_free_energy:.3f} kcal/mol')
    return free_energy, d_free_energy

def visualize_latent_space(latent_space: torch.Tensor, molecules: list[str]):
    """
    Visualize the latent space representation of a set of molecules
    :param latent_space: The Nx2 dimenesional encoding values of the N molecules
    :param molecules: The list of molecule labels used for annotations of the scatter plot.
    """
    """ Scatter plot of all encoded molecules """
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(latent_space[:, 0], latent_space[:, 1], s=15, color='red')
    for n in range(len(molecules)):
        ax.annotate(str(molecules[n]), (latent_space[n, 0], latent_space[n, 1]), fontsize=6)
    ax.set(xlabel="Encoding dimension 1", ylabel="Encoding dimension 2")
    """ Visualize encoding space densities using KDE plots """
    ax_right = fig.add_axes([0.901, 0.11, 0.04, ax.get_position().height])
    ax_top = fig.add_axes([0.125, 0.88, ax.get_position().width, 0.04])
    sns.kdeplot(y=latent_space[:, 1], ax=ax_right,
                color='gray', fill=True, linewidth=0.2, bw_adjust=0.5)
    sns.kdeplot(x=latent_space[:, 0], ax=ax_top,
                color='gray', fill=True, linewidth=0.2, bw_adjust=0.5)
    ax_right.set(xlabel=None, ylabel=None, ylim=ax.get_ylim())
    ax_right.axis('off')
    ax_top.set(xlabel=None, ylabel=None, xlim=ax.get_xlim())
    ax_top.axis('off')
    plt.show()

def acquisition_function(values: torch.Tensor, uncertainty: torch.Tensor,
                         best_known_value: float, xi: float = 0.0) -> torch.Tensor:
    """
    The acquisition function implemented here is the expected improvement. See this
    link for an explanation: https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html#expected-improvement-ei
    :param values: Mean prediction values from the surrogate model.
    :param uncertainty: Predicted standard deviation from the surrogate model.
    :param best_known_value: Best so far observed value.
    :param best_known_value: Best so far observed value.
    """
    z = (values - best_known_value - xi)
    return z * norm.cdf(z / uncertainty) + uncertainty * norm.pdf(z / uncertainty)

def argmax_width_excluded_indices(values: np.ndarray, excluded_indices: list[int]) -> int:
    """
    Find the index of the maximum value in an array, excluding specified indices. If 
    there are multiple maximum values, one of them is randomly selected. The numpy
    masked array is used to handle the exclusion of indices.
    :param values: A numpy array of values.
    :param excluded_indices: A list of indices to exclude from the search for the maximum.
    :returns: The index of the maximum value, excluding specified indices.
    """
    if len(values) == 0:
        raise ValueError("The input array is empty.")
    mask = np.isin(np.arange(len(values)), list(excluded_indices))
    values = np.ma.masked_array(values, mask=mask)
    return np.random.choice(np.nonzero(values == values.max())[0])

class SurrogateModel:

    """
    A surrogate model for predicting target values based on a latent space representation.
    This model uses a Gaussian Process Regressor with a radial basis function kernel.
    It is designed to fit a set of data points and predict values for the latent space.
    """

    def __init__(self, latent_space: np.ndarray):
        """
        Initialize the surrogate model with a latent space representation. The latent space
        is used for all predictions of the model.
        :param latent_space: A Nx2 numpy array representing the latent space of N molecules.
        """
        kernel = RBF(length_scale=0.5, length_scale_bounds=(0.05, 2))
        self.gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=9, alpha=0.05
        )
        self.latent_space = latent_space

    def fit(self, data: dict[int, float]):
        """
        Fit the surrogate model to the provided data. The data should be a dictionary
        where keys are indices of the latent space and values are the target values.
        :param data: A dictionary with keys as indices of the latent space and values as
            target values.
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        if len(data) == 0:
            return
        X = [self.latent_space[i] for i in data.keys()]
        Y = list(data.values())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gaussian_process.fit(X, Y)

    def predict(self):
        """
        Predict the target values for the latent space using the fitted surrogate model.
        The function does not take arguments, as it uses the latent space provided during
        the initialization of the model for predictions.
        :returns: A tuple containing the predicted values and their standard deviations.
        """
        return self.gaussian_process.predict(self.latent_space, return_std=True)
    
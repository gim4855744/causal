"""Based on https://github.com/loeweX/AmortizedCausalDiscovery (MIT License)."""

import numpy as np
from scipy.integrate import ode
from tqdm import tqdm

__all__ = ['generate_kuramoto']


class Kuramoto:

    """
    Implementation of Kuramoto coupling model[1] with harmonic terms and possible perturbation.
    It uses NumPy and Scipy's implementation of Runge-Kutta 4(5) for numerical integration.

    Usage example:
    >>> kuramoto = Kuramoto(initial_values)
    >>> phase = kuramoto.solve(X)

    [1] Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence (Vol. 19). doi: doi.org/10.1007/978-3-642-69689-3
    """

    _noises = {
        'logistic': np.random.logistic,
        'normal': np.random.normal,
        'uniform': np.random.uniform,
        'custom': None
    }
    noise_types = _noises.keys()

    def __init__(self, init_values, noise=None):
        
        """
        Passed arguments should be a dictionary with NumPy arrays for initial phase (Y0), intrisic frequencies (W) and coupling matrix (K).
        """

        self.dtype = np.float32

        self.dt = 1.
        self.init_phase = np.array(init_values['Y0'])
        self.W = np.array(init_values['W'])
        self.K = np.array(init_values['K'])

        self.n_osc = len(self.W)
        self.m_order = self.K.shape[0]

        self.noise = noise

    @property
    def noise(self):
        """
        Sets perturbations added to the system at each timestamp.
        Noise function can be manually defined or selected from predefined by assgining corresponding name. List of available pertrubations is reachable through `noise_types`.
        """
        return self._noise
    
    @noise.setter
    def noise(self, _noise):

        self._noise = None
        self.noise_params = None
        self.noise_type = 'custom'

        # If passed a function.
        if callable(_noise):
            self._noise = _noise

        # In case passing string.
        elif isinstance(_noise, str):

            if _noise.lower() not in self.noise_types:
                self.noise_type = None
                raise NameError("No such noise method")

            self.noise_type = _noise.lower()
            self.update_noise_params(self.dt)

            noise_function = self._noises[self.noise_type]
            self._noise = lambda: np.array([noise_function(**p) for p in self.noise_params])

    def update_noise_params(self, dt):

        self.scale_func = lambda dt: dt / np.abs(self.W ** 2)
        scale = self.scale_func(dt)

        if self.noise_type == 'uniform':
            self.noise_params = [{'low': -s, 'high': s} for s in scale]
        elif self.noise_type in self.noise_types:
            self.noise_params = [{'loc': 0, 'scale': s} for s in scale]
        else:
            pass

    def kuramoto_ODE(self, t, y, arg):

        """
        General Kuramoto ODE of m'th harmonic order.
        Argument `arg` = (w, k), with
            w -- iterable frequency
            k -- 3D coupling matrix, unless 1st order
        """

        w, k = arg
        yt = y[:, None]
        dy = y - yt
        phase = w.astype(self.dtype)
        if self.noise != None:
            n = self.noise().astype(self.dtype)
            phase += n
        for m, _k in enumerate(k):
            phase += np.sum(_k * np.sin((m + 1) * dy), axis=1)

        return phase

    def kuramoto_ODE_jac(self, t, y, arg):

        """Kuramoto's Jacobian passed for ODE solver."""

        w, k = arg
        yt = y[:, None]
        dy = y - yt

        phase = [m * k[m - 1] * np.cos(m * dy) for m in range(1, 1 + self.m_order)]
        phase = np.sum(phase, axis=0)

        for i in range(self.n_osc):
            phase[i, i] = -np.sum(phase[:, i])

        return phase

    def solve(self, t):

        """Solves Kuramoto ODE for time series `t` with initial parameters passed when initiated object."""

        dt = t[1] - t[0]
        if self.dt != dt and self.noise_type != 'custom':
            self.dt = dt
            self.update_noise_params(dt)

        kODE = ode(self.kuramoto_ODE, jac=self.kuramoto_ODE_jac)
        kODE.set_integrator("dopri5")

        # Set parameters into model.
        kODE.set_initial_value(self.init_phase, t[0])
        kODE.set_f_params((self.W, self.K))
        kODE.set_jac_params((self.W, self.K))

        if self._noise != None:
            self.update_noise_params(dt)

        phase = np.empty((self.n_osc, len(t)))

        # Run ODE integrator.
        for idx, _t in enumerate(t[1:]):
            phase[:,idx] = kODE.y
            kODE.integrate(_t)

        phase[:, -1] = kODE.y

        return phase


def simulate_kuramoto(n_atoms, n_timesteps, T, dt):
    
    intrinsic_freq = np.random.rand(n_atoms) * 9 + 1
    initial_phase = np.random.rand(n_atoms) * 2 * np.pi
    edges = np.random.choice(2, size=(n_atoms, n_atoms), p=[0.5, 0.5])
    np.fill_diagonal(edges, 0)

    kuramoto = Kuramoto({'W': intrinsic_freq, 'K': np.expand_dims(edges, 0), 'Y0': initial_phase})

    odePhi = kuramoto.solve(T)

    # Subsample.
    phase_diff = np.diff(odePhi)[:, ::10] / dt
    trajectories = np.sin(odePhi[:, :-1])[:, ::10]

    # Normalize dPhi (individually).
    min_vals = np.expand_dims(phase_diff.min(1), 1)
    max_vals = np.expand_dims(phase_diff.max(1), 1)
    phase_diff = (phase_diff - min_vals) * 2 / (max_vals - min_vals) - 1

    # Get absolute phase and normalize.
    phase = odePhi[:, :-1][:, ::10]
    min_vals = np.expand_dims(phase.min(1), 1)
    max_vals = np.expand_dims(phase.max(1), 1)
    phase = (phase - min_vals) * 2 / (max_vals - min_vals) - 1

    # If oscillator is uncoupled, set trajectory to dPhi to 0 for all t.
    isolated_idx = np.where(edges.sum(1) == 0)[0]
    phase_diff[isolated_idx] = 0.

    # Normalize frequencies to [-1, 1].
    intrinsic_freq = (intrinsic_freq - 1.) * 2 / (10. - 1.) - 1.

    phase_diff = np.expand_dims(phase_diff, -1)[:, :n_timesteps, :]
    trajectories = np.expand_dims(trajectories, -1)[:, :n_timesteps, :]
    phase = np.expand_dims(phase, -1)[:, :n_timesteps, :]
    intrinsic_freq = np.expand_dims(np.repeat(np.expand_dims(intrinsic_freq, -1), n_timesteps, axis=1), -1)

    feats = np.concatenate((phase_diff, trajectories, phase, intrinsic_freq), -1)

    return feats, edges


def generate_kuramoto(
    n_sims=50000,
    n_atoms=5,
    length=10000,
    sample_freq=100
):
    
    """
    Generate Kuramoto simulations dataset.
    The final length of each simulation trajectory will be 'length // sample_freq'.

    Parameters
    ----------
        n_sims: int
            Number of simulations to generate.
        n_atoms: int
            Number of atoms (i.e., time-series) in the simulation.
        length: int
            Length of each simulation trajectory.
        sample_freq: int
            Sampling frequency of the trajectory.

    Returns
    -------
        feats: np.ndarray
            Simulated time-series of shape (n_sims, n_atoms, length // sample_freq, 4).
            4 indicates the four features: phase difference, trajectory, absolute phase, and intrinsic frequency.
        edges: np.ndarray
            Causal edges (binary) of shape (n_sims, n_atoms, n_atoms).
    """
    
    n_timesteps = length // sample_freq
    t0, t1, dt = 0, length // sample_freq // 10, 0.01
    T = np.arange(t0, t1, dt)
    feats_all , edges_all  = [], []

    for _ in tqdm(range(n_sims)):
        feats, edges = simulate_kuramoto(n_atoms, n_timesteps, T, dt)
        feats_all.append(feats)
        edges_all.append(edges)

    feats = np.array(feats_all)
    edges = np.array(edges_all)

    return feats, edges

"""Based on https://github.com/loeweX/AmortizedCausalDiscovery (MIT License)."""

import numpy as np
from tqdm import tqdm

__all__ = ['generate_springs']


class SpringSim:

    def __init__(self, n_balls, interaction_strength, box_size=5.0, loc_std=0.5, vel_norm=0.5, noise_var=0.0):

        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0.0, 0.5, 1.0])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _clamp(self, loc, vel):

        """
        Params:
        loc: 2xN location at one time stamp
        vel: 2xN velocity at one time stamp
        Returns:
        location and velocity after hitting walls and returning after elastically colliding with walls
        """

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_edges(self, spring_prob):
        edges = np.random.choice(self._spring_types, size=(self.n_balls, self.n_balls), p=spring_prob)
        np.fill_diagonal(edges, 0)
        return edges

    def get_edges(self, spring_prob=[0.5, 0, 0.5]):
        edges=self.sample_edges(spring_prob)
        return edges

    def sample_trajectory(
        self,
        T=10000,
        sample_freq=10,
        spring_prob=[0.5, 0, 0.5]
    ):
        
        n = self.n_balls
        T_save = T // sample_freq
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        
        edges = self.get_edges(spring_prob=spring_prob)

        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm

        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):

            forces_size = -self.interaction_strength * edges
            np.fill_diagonal(forces_size, 0)  # self forces are zero (fixes division by zero)
            F = (
                forces_size.reshape(1, n, n) * np.concatenate((
                    np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(1, n, n),
                    np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(1, n, n))
                )
            ).sum(axis=-1)  # sum over influence from different particles to get their joint force
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F

            # run leapfrog
            for i in range(1, T):

                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = -self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)

                F = (forces_size.reshape(1, n, n) * np.concatenate((
                    np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(1, n, n),
                    np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(1, n, n)))
                ).sum(axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F

            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var

            return loc, vel, edges


def generate_springs(
    n_sims=50000,
    n_balls=5,
    length=10000,
    sample_freq=100
):

    """
    Generate Springs simulations dataset.
    The final length of each simulation trajectory will be 'length // sample_freq'.

    Parameters
    ----------
    n_sims: int
        Number of simulations to generate.
    n_balls: int
        Number of balls (i.e., time-series) in the simulation.
    length: int
        Lnegth of each simulation trajectory.
    sample_freq: int
        Sampling frequency of the simulation trajectory.

    Returns
    -------
    feats: np.ndarray
        Array of shape (n_sims, n_balls, length // sample_freq, 4).
        4 indicates the four features: location and velocity in x and y dimensions.
    edges: np.ndarray
        Causal edges (binary) of shape (n_sims, n_balls, n_balls).
    """

    sim = SpringSim(n_balls, interaction_strength=0.1)
    loc_all, vel_all, edges_all = [], [], []

    for i in tqdm(range(n_sims)):
        loc, vel, edges = sim.sample_trajectory(T=length, sample_freq=sample_freq)
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    feats = np.concatenate([loc_all, vel_all], axis=2).transpose(0, 3, 1, 2)
    edges = np.stack(edges_all)

    return feats, edges

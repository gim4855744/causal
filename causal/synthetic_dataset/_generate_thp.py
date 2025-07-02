import numpy as np
from castle.datasets import Topology, DAG, THPSimulation

__all__ = ['generate_thp']


def generate_thp(n_devices=40, n_events=20, device_sparsity=0.05, event_sparsity=0.1):

    """
    Generate synthetic Topological Hawkes Process (THP) simulations.

    Parameters
    ----------
    n_devices: int
        Number of devices in the simulation.
    n_events: int
        Number of events in the simulation.
    device_sparsity: float
        Sparsity of the device topology, between 0 and 1.
    event_sparsity: float
        Sparsity of the event topology, between 0 and 1.

    Returns
    -------
    device_edges: np.ndarray
        Array of device edges representing the device topology.
    event_edges: np.ndarray
        Array of event edges representing the event topology.
    x: pd.DataFrame
        DataFrame containing the simulated events.
    """

    n_edges = int(n_devices * n_devices * device_sparsity)
    device_edges = np.array(Topology.erdos_renyi(n_devices, n_edges))

    n_edges = int(n_events * n_events * event_sparsity)
    event_edges = np.array(DAG.erdos_renyi(n_events, n_edges))

    simulator = THPSimulation(event_edges, device_edges, mu_range=(0.00003, 0.00005), alpha_range=(0.02, 0.03))
    x = simulator.simulate(T=10000, max_hop=2)
    x = x.rename(columns={'node': 'device', 'timestamp': 'start_timestamp'})
    x['end_timestamp'] = x['start_timestamp'] + x['duration']
    x = x.drop(columns=['duration'], axis=1)
    x = x.astype(int)

    return device_edges, event_edges, x

"""Strategy Module.

Provides different strategies that agents can use.
"""

from math import exp
from random import random, getrandbits, choice
from agent import MigrationAgent


def random_move(
    agent: MigrationAgent, influencers: list[MigrationAgent], beta: float
) -> bool:
    """Makes a random move (either cooperate/defect).

    Returns
    ------
    bool
        the move to be made by the agent
    """

    return bool(getrandbits(1))


def conditionally_imitate(
    agent: MigrationAgent, influencers: list[MigrationAgent], beta: float
) -> bool:
    """Conditionally imitate a more successful neighbour.

    Parameters
    ----------
    agent : MigrationAgent
        the agent that should make the moave
    influencers : list[MigrationAgent]
        the other agents that the passed agent can conditionally imitate
    beta : float
        a value between 0 and 1 for the BETA parameter of the fermi function

    Returns
    ------
    bool
        the move to be made by the agent
    """

    # if this is the first step, make a random move
    if agent.status.last_move is None:
        return bool(getrandbits(1))

    # if there are no influencers, repeat last move
    if len(influencers) == 0:
        return agent.status.last_move

    # select a random neighbour to conditionally imitate
    other = choice(influencers)

    # fermi probability for updating strategy
    fermi_prob = 1.0 / (
        1 + exp(-1 * beta * (other.status.payoff - agent.status.payoff))
    )

    # conditionally copy the other agent's last move or repeat the last move
    return other.status.last_move if random() < fermi_prob else agent.status.last_move

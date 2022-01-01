"""payoff calculators for different social dilemma games."""

from math import ceil
from enum import Enum


# TODO: complete


class GameType(Enum):
    """Enum for different types of dilemma games"""

    NIPD = "nipd"
    NISD = "nisd"
    PGG = "pgg"


class Game:
    """catalog of different games that can be played.

    Attributes
    ----------
    game_type: GameType
        the type of game this object represents

    Methods
    -------
    calculate_payoff: float
        calculate and return the payoff for the specified game type.
    """

    def __init__(self, game_type: GameType) -> None:
        self.game_type = game_type

        self._payoff_calculators = {
            GameType.NIPD: self._nipd,
            GameType.NISD: self._nisd,
            GameType.PGG: self._pgg,
        }

        self._payoff_range_calculators = {
            GameType.NIPD: self._nipd_range,
            GameType.NISD: self._nisd_range,
            GameType.PGG: self._pgg_range,
        }

    def calculate_payoff(
        self,
        cost: float,
        num_cooperators: int,
        population: int,
        move: bool,
        threshold: float = 0,
    ) -> float:
        """calculate and return payoff based on the game type.

        Parameters
        ----------
        cost : float
            cost of defection (between 0 and 1; benefit is always 1)
        num_cooperators : int
            total number of cooperators in the group
        population : int
            population of the group
        move : bool
            the agent's move; true: cooperation, false: defection
        threshold : float
           the  ratio of population that must cooperate for benefits to be acheived

        Returns
        -------
        float
            the agent's payoff
        """

        assert 0 <= cost <= 1.0, "cost must be in the range 0-1"
        assert num_cooperators >= 0, "number of cooperators must be > 0"
        assert population >= 0, "population must be > 0"
        assert 0 <= threshold <= 1.0, "threshold must be in the range 0-1"

        assert (
            num_cooperators <= population
        ), "number of cooperators cannot be more than the population"

        assert not (
            move and num_cooperators == 0
        ), "number of cooperators cannot be zero when the agent has cooperated"

        return self._payoff_calculators[self.game_type](
            cost, num_cooperators, population, move, threshold
        )

    @staticmethod
    def _nipd(
        cost: float,
        num_cooperators: int,
        population: int,
        move: bool,
        threshold: float = 0,
    ) -> float:
        """Calculate payoff for and agent in the NIPD game.

        Parameters
        ----------
        cost : float
            cost of defection (between 0 and 1; benefit is always 1)
        num_cooperators : int
            total number of cooperators in the group
        population : int
            population of the group
        move : bool
            the agent's move; true: cooperation, false: defection
        threshold : float
           the  ratio of population that must cooperate for benefits to be acheived

        Returns
        -------
        float
            the agent's payoff
        """

        # if there are no cooperators, then no payoff
        # if num_cooperators == 0:
        #    return 0

        # calculate the min num. of cooperators required to meet the threshold
        # if there is no threshold, its, zero, so min_cooperators = 0
        min_cooperators = ceil(population * threshold)

        # when threshold is not met
        if num_cooperators < min_cooperators:
            # defectors get 0 payoff, cooperators are punished.
            return -cost * (population - 1) if move else 0

        # if there are no cooperators, there's no payoff
        if num_cooperators == 0:
            return 0

        # if the agent is a cooperator, then it can benefit from the actions of
        # other cooperators only
        num_cooperators -= move

        # threshold is met, benefit is always 1
        return num_cooperators - (move * cost * (population - 1))

    @staticmethod
    def _nisd(
        c: float,
        n: int,
        N: int,
        m: bool,
        t: float = 0,
    ) -> float:
        """Calculate payoff for and agent in the NISD game.

        Parameters
        ----------
        c : float
            cost of defection (between 0 and 1; benefit is always 1)
        n : int
            total number of cooperators in the group
        N : int
            population of the group
        m : bool
            the agent's move; true: cooperation, false: defection
        t : float
           the  ratio of population that must cooperate for benefits to be acheived

        Returns
        -------
        float
            the agent's payoff
        """

        # when threshold is not met
        # defectors get 0 payoff, cooperators are punished.
        # Souza et. al, 'Evolution of cooperation under N-person snowdrift games', 2009
        if t > 0 and n < (k := ceil(t * N)):
            return -m * c / k

        return 1 - m * (c / n) if n > 0 else 0

    @staticmethod
    def _pgg(
        c: float,
        n: int,
        N: int,
        m: bool,
        t: float = 0,
    ) -> float:
        """Calculate payoff for an agent in the public goods game.

        Parameters
        ----------
        c : float
            cost of defection (between 0 and 1; benefit is always 1)
        n : int
            total number of cooperators in the group
        N : int
            population of the group
        m : bool
            the agent's move; true: cooperation, false: defection
        t : float
           the  ratio of population that must cooperate for benefits to be acheived

        Returns
        -------
        float
            the agent's payoff
        """

        # calculate the min num. of cooperators required to meet the threshold
        # if there is no threshold, its, zero, so min_cooperators = 0
        k = ceil(N * t)

        # when threshold is not met
        if n < k:
            # defectors get 0 payoff, cooperators are punished
            return -c * (N - 1) / k if m else 0

        # if there are no cooperators, there's no payoff
        if n == 0:
            return 0

        # if the agent is a cooperator, then it can benefit from the actions of
        # other cooperators only
        n -= m

        # threshold is met, benefit is always 1
        return n - (m * c * (N - 1) / (n + 1))

    def payoff_range(
        self, cost: float, threshold: float, population: int
    ) -> (float, float):
        """Calculate min/max payoff for the given population size.

        Parameters
        ----------
        cost : float
            cost of defection (between 0 and 1; benefit is always 1)
        threshold : float
            the ratio of population that must cooperate for benefits to be acheived
        population : int
            number of agents in the population

        Returns
        -------
        (float, float)
            min and max payoffs possible for the population
        """

        return self._payoff_range_calculators[self.game_type](
            cost, population, threshold
        )

    def _nipd_range(self, c: float, N: int, t: float) -> (float, float):
        """Calculate min/max payoff for the given population size"""

        return

    def _nisd_range(self, c: float, N: int, th: float) -> (float, float):
        """Calculate min/max payoff for the given population size"""

        return (
            # min payoff: when threshold is not met, and the agent is the only coopreator
            min(self._nisd(c, 1, N, True, th), self._nisd(c, 0, N, False, th)),
            # max payoff for a defecting agent when others are cooperators
            self._nisd(c, N - 1, N, False, th),
        )

    def _pgg_range(self, c: float, N: int, t: float) -> (float, float):
        """Calculate min/max payoff for the given population size"""

        return

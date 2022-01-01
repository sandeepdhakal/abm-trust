"""Migration model."""

from __future__ import annotations
from random import random, choice, getrandbits
from dataclasses import dataclass
from typing import Tuple, Dict, List
from collections import Counter
from math import exp
from mesa import Model, Agent
from mesa.space import NetworkGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
from networkx import Graph

BETA = 1.0


def percent_cooperators(model):
    """Return the percentage of cooperators in the model."""

    return (model.log.num_cooperators * 100) / model.num_agents


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class ModelConfig:
    """Class for storing the migration model's configuration."""

    num_agents: int = 1000
    num_groups: int = 100
    num_tags: int = 2
    social_network: bool = False
    migrate: bool = False
    trust_threshold: float = 0
    mutation_prob: float = 0
    max_group_size: int = 1000
    migration_wait_period: int = 0


class Tag:
    """Class for storing information about a tag."""

    __slots__ = [
        "tag_id",
        "count",
        "cooperators",
        "migrators",
        "trustworthiness",
    ]

    def __init__(self, tag_id):
        self.tag_id = tag_id
        self.count = self.cooperators = self.migrators = 0
        self.trustworthiness = 0

    def reset_timestep_stats(self):
        """Reset timestep specific stats."""
        self.cooperators = self.migrators = 0
        self.trustworthiness = 0

    def update_stats(self, agent):
        """Update the tag's stats based on the agent's stats."""
        self.cooperators += agent.status.last_move
        self.migrators += agent.status.migrated_in_ts

    def as_dict(self):
        """Return the tag's information as a dict."""
        return {
            "id": self.tag_id,
            "count": self.count,
            "cooperators": self.cooperators,
            "migrators": self.migrators,
            "trustworthiness": self.trustworthiness,
        }

    @staticmethod
    def set_trustworthiness(
        tags: List[Tag], tag_trusts: Dict[str, float], num_agents: int
    ):
        """Set the trustworthiness of each tag by averaging the total trust in it."""
        for k, tag in tags.items():
            tag.trustworthiness = tag_trusts[k] / num_agents


class MigrationModelStats:
    """Class for storing the migration model's stats."""

    __slots__ = [
        "num_migrators",
        "num_cooperators",
        "global_trust",
        "total_payoff",
        "cooperator_migrators",
        "migration_attempts",
        "tags_in_groups",
    ]

    def __init__(self):
        self.num_migrators: int = 0
        self.num_cooperators: int = 0
        self.global_trust: float = 0.0
        self.total_payoff: float = 0.0
        self.cooperator_migrators: int = 0
        self.migration_attempts: int = 0
        self.tags_in_groups: Dict[str, np.array] = {}

    def reset_timestep_logs(self, tags, num_groups):
        """Reset the log items that are timestep dependent."""
        self.num_migrators = 0
        self.num_cooperators = 0
        self.cooperator_migrators = 0
        self.migration_attempts = 0
        self.total_payoff = 0
        self.tags_in_groups = {k: np.zeros((num_groups,), dtype=int) for k in tags}

    def update_with_agent_status(self, agent):
        """Update the status based on the passed agent."""
        self.num_cooperators += agent.status.last_move
        self.total_payoff += agent.status.payoff
        self.migration_attempts += agent.status.attempted_migration_in_ts

        if agent.status.migrated_in_ts:
            self.num_migrators += 1
            self.cooperator_migrators += agent.status.last_move

    def update_tags_in_groups(self, group):
        """Update the number of each tag in each group."""
        for tag, count in group.tag_counter.items():
            self.tags_in_groups[tag][group.unique_id] = count


# pylint: disable=too-many-instance-attributes
class MigrationModel(Model):
    """A model with a fixed number of agents in a number of groups."""

    # pylint: disable=super-init-not-called,too-many-arguments
    def __init__(
        self,
        config: ModelConfig,
        social_network: Graph,
        payoff_calculator,
        payoff_range_calculator,
        trust_updater,
    ):
        """Init MigrationModel with the given configuration."""

        self.config = config

        # Barabasi Albert graph as the network structure.
        self.grid = NetworkGrid(social_network)
        self.schedule = RandomActivation(self)
        self.log = MigrationModelStats()
        self.payoff_calculator = payoff_calculator
        self.payoff_range_calculator = payoff_range_calculator
        self.trust_updater = trust_updater

        # create tags
        self.tags = {f"t{i}": Tag(f"t{i}") for i in range(config.num_tags)}

        # create groups
        self.groups = [
            Group(unique_id=i, model=self, max_size=self.config.max_group_size)
            for i in range(self.config.num_groups)
        ]
        self.__create_agents()

        # remove any groups that don't have any members
        self.groups = [x for x in self.groups if len(x.members) > 1]

        self.__setup_datacollectors()

    def __create_agents(self):
        """Create agents for the model"""

        # NOTE: group size is always an integer
        group_size = self.config.num_agents // self.config.num_groups

        # Create agents
        tag_ids = list(self.tags.keys())
        for i, node in enumerate(self.grid.G.nodes()):
            tag_id = choice(tag_ids)
            _agent = MigrationAgent(i, self, tag_id)
            self.schedule.add(_agent)
            self.tags[tag_id].count += 1

            # Add the agent to the node
            self.grid.place_agent(_agent, node)

            # Add the agent to a random group
            group = self.groups[i // group_size]
            group.add_agent(_agent)

        if self.config.social_network:
            for agent in self.schedule.agents:
                agent.friends = self.get_network_connections(agent.pos)

    def __setup_datacollectors(self):
        """Setup datacollectors for collecting information about the model and its
        agents"""

        group_ids = list(range(self.config.num_groups))
        tag_ids = self.tags.keys()
        tables = {
            "group": group_ids,
            "tag": tag_ids,
            "tags_in_groups": tag_ids,
        }
        self.datacollector = DataCollector(
            model_reporters={
                "cooperators": lambda m: m.log.num_cooperators,
                "migrators": lambda m: m.log.num_migrators,
                "trust": lambda m: m.log.global_trust,
                "total_payoff": lambda m: m.log.total_payoff,
                "cooperator_migrators": lambda m: m.log.cooperator_migrators,
                "migration_attempts": lambda m: m.log.migration_attempts,
            },
            tables=tables,
        )

    def step(self):
        """Advance the model by one step."""
        # reset all logs that might change in each timestep
        self.log.reset_timestep_logs(self.tags.keys(), len(self.groups))

        # ask all the agents to move to the next step
        self.schedule.step()

        # Update the groups' status once all agents have made their move
        for group in self.groups:
            group.step()

        # update the agents, and collect stats
        # first, reset the timestep related stats
        cumulative_global_trust: float = 0.0
        group_trust = np.zeros(self.config.num_groups, dtype=float)
        tag_trusts: Dict[str, float] = dict.fromkeys(self.tags.keys(), 0.0)
        for _, tag in self.tags.items():
            tag.reset_timestep_stats()

        for agent in self.schedule.agents:
            # each agent has to update once its graup has updated
            agent.update()

            # update the trust values
            cumulative_global_trust += agent.status.trust_in_group
            group_trust[agent.group.unique_id] += agent.status.trust_in_group
            for tag, trust in agent.status.trust_in_tags.items():
                tag_trusts[tag] += trust

            # attempt migration only if the model allows migration.
            if self.config.migrate:
                agent.attempt_migration()

            # now, update the log
            self.log.update_with_agent_status(agent)

            # and the tag stats
            self.tags[agent.tag].update_stats(agent)

        # calculate and log the global trust
        self.log.global_trust = cumulative_global_trust / self.config.num_agents

        for group in self.groups:
            group.trustworthiness = group_trust[group.unique_id] / len(group)
            self.log.update_tags_in_groups(group)

        # calculate and log the average turstworthiness of each tag
        Tag.set_trustworthiness(self.tags, tag_trusts, self.config.num_agents)

    def collect_data(self):
        """Collect model/agent data."""
        # Collect data for the timestep.
        self.datacollector.collect(self)

        # about tags
        self.datacollector.add_table_row(
            "tag",
            {k: [*v.as_dict().values()] for k, v in self.tags.items()},
        )

        # about groups
        self.datacollector.add_table_row(
            "group",
            {
                k: [len(v), v.percent_cooperators, v.trustworthiness]
                for k, v in enumerate(self.groups)
            },
        )

        # about tags in groups
        self.datacollector.add_table_row("tags_in_groups", self.log.tags_in_groups)

    def get_network_connections(self, pos: Tuple[int, int]):
        """Return the network connections of the agent at the specified position."""
        friend_nodes = self.grid.get_neighbors(pos, include_center=False)
        return self.grid.get_cell_list_contents(friend_nodes)

    def can_agent_migrate(self, agent: MigrationAgent) -> bool:
        """Checks whether an agent is allowed to migrate in the current timestep

        Parameters
        ----------
        agent : MigrationAgent
            the agent for which the check is to be done

        Returns
        -------
        bool
            whether the passed agent can migrate
        """

        # an agent cannot migrate within a specified number of timesteps since last
        # migration
        if agent.status.timesteps_since_migration <= self.config.migration_wait_period:
            return False

        # NOTE: if current group has 2 or less members, forbid migration.
        if len(agent.group) <= 2:
            return False

        return True


# pylint: disable=too-few-public-methods,too-many-instance-attributes
class MigrationAgentStatus:
    """Class for storing the migration agent's status."""

    __slots__ = [
        "trust_in_tags",
        "trust_in_group",
        "_payoff",
        "_cumulative_payoff",
        "payoff_diff",
        "move",
        "last_move",
        "migrated_in_ts",
        "attempted_migration_in_ts",
        "total_migrations",
        "timesteps_since_migration",
    ]

    def __init__(
        self,
        payoff: float,
        move: bool,
        last_move: bool,
        trust_in_tags: Dict[str, float] = None,
    ):
        self.trust_in_tags = trust_in_tags
        self._payoff = payoff
        self._cumulative_payoff = payoff
        self.move = move
        self.last_move = last_move

        self.trust_in_group = 0
        self.payoff_diff = 0
        self.migrated_in_ts = False
        self.attempted_migration_in_ts = False
        self.total_migrations = 0
        self.timesteps_since_migration = 0

    @property
    def payoff(self):
        """The agent's current payoff."""
        return self._payoff

    @payoff.setter
    def payoff(self, new_payoff):
        """Set the agent's payoff and also calculate the difference in payoff
        compared to the previous timestep."""
        self._cumulative_payoff += new_payoff
        self.payoff_diff = new_payoff - self._payoff
        self._payoff = new_payoff

    @property
    def cumulative_payoff(self):
        """The agent's total payoff so far."""
        return self._cumulative_payoff

    def update_stats_for_migration(self):
        """Update the migration related status of an agent that has migrated."""

        self.migrated_in_ts = True
        self.timesteps_since_migration = 0
        self.total_migrations += 1


class MigrationAgent(Agent):
    """An agent that can migrate."""

    __slots__ = [
        "tag",
        "status",
        "group",
        "friends",
        "neighbours",
        "migration_strategy",
    ]

    def __init__(self, unique_id, model, tag):
        """Init MigrationAgent with unique id and model."""
        super().__init__(unique_id, model)

        # the label seen by other agents
        self.tag = tag

        # current status
        self.status = MigrationAgentStatus(payoff=0, move=None, last_move=None)

        # the agent's physical/geographical group.
        self.group = None

        # the agent's friends.
        # if the model allows social networks, the friends are the connected
        # nodes, otherwise, they are members of the physical group.
        self.friends: List

        # the agent's neighbours in the physical group.
        self.neighbours = None

        # the agent's migration strategy
        self.migration_strategy = None

    def as_dict(self):
        """Return the agent's information as a dict."""
        return {
            "id": self.unique_id,
            "total payoff": self.status.cumulative_payoff,
            "total migrations": self.status.total_migrations,
            "trust in group": self.status.trust_in_group,
        }

    @staticmethod
    def make_move(agent, neighbours):
        """Conditionally follow a more successful neighbour."""

        # if this is the first step, make a random move
        if agent.status.last_move is None:
            return bool(getrandbits(1))

        # if there are no neighbours, repeat last move
        if len(neighbours) == 0:
            return agent.status.last_move

        # select a random neighbour to conditionally imitate
        other = choice(neighbours)

        # fermi probability for updating strategy
        fermi_prob = 1.0 / (
            1 + exp(-1 * BETA * (other.status.payoff - agent.status.payoff))
        )

        # conditionally copy the other agent's last move or
        return (
            other.status.last_move if random() < fermi_prob else agent.status.last_move
        )

    def step(self):
        """Actions for each timestep."""
        # initialise the agent's group members
        self.neighbours = set(self.group.members) - {self}

        # Initialise the list of friends
        # friends are the connected nodes if social network is turned on
        # otherwise they are members of the physical group.
        influencers = (
            list(set(self.friends) | self.neighbours)
            if self.model.config.social_network
            else list(self.neighbours)
        )

        # Ask the strategy to make its move depending on the influencers.
        # The move can be switched with a certain probability.
        move = MigrationAgent.make_move(self, influencers)
        if random() < self.model.config.mutation_prob:
            move = not move
        self.status.move = move

    def update(self):
        """Update the agent's status at the end of the timestep."""
        self.status.last_move = self.status.move

        # get the trust in the current group
        self.status.trust_in_group = self.trust_in_group(self.group)

        if not self.status.migrated_in_ts:
            self.status.timesteps_since_migration += 1

    def update_trust(self, tag_counter):
        """Update the trust values of this agent in other tags."""
        total_neighbours = sum(tag_counter.values())
        trust_updater = self.model.trust_updater
        trusts = self.status.trust_in_tags
        payoff = self.status.payoff

        for tag in tag_counter:
            trusts[tag] = trust_updater(
                trusts[tag], payoff, tag_counter[tag], total_neighbours
            )

    def trust_in_group(self, group) -> float:
        """Calculates the trust in a particular group for this agent."""
        own_group = self.group.unique_id == group.unique_id
        tag_counter = group.tag_counter - Counter({self.tag: 1 if own_group else 0})
        trust_in_tags = self.status.trust_in_tags
        return sum((trust_in_tags[t] * tag_counter[t] for t in tag_counter)) / sum(
            tag_counter.values()
        )

    def get_migration_options(self) -> list[Group]:
        """Returns the list of groups that this agent can migrate to."""

        # NOTE: assuming that agent wants to migrate to friends' groups.
        # NOTE: assuming that there are no barriers to migrating to group
        friends_groups = (
            {x.group for x in self.friends}
            if self.model.config.social_network
            else set(self.model.groups)
        )
        friends_groups = friends_groups - {self.group}
        return list(friends_groups)

    def attempt_migration(self) -> bool:
        """Move the agent from the current group to a new group."""

        # reset the migration status for this timestep
        self.status.migrated_in_ts = False
        self.status.attempted_migration_in_ts = False

        # check with the model if this agent is allowed to migrate
        if not self.model.can_agent_migrate(self):
            return False

        # if the trust in the group is more than the threshold, don't migrate
        if self.status.trust_in_group >= self.model.config.trust_threshold:
            return False

        # at this point, we know that the agent wants to migrate.
        self.status.attempted_migration_in_ts = True

        # get the migration destination based on migration strategy
        dest = self.migration_strategy.get_migration_destination(self)
        if dest is not None:
            migrate_agent(self, self.group, dest)
            return True

        return False


class Group(Agent):
    """A group of agents."""

    def __init__(self, unique_id, model, max_size):
        """Init a Group."""
        super().__init__(unique_id, model)

        self.members: set = set()
        self.num_cooperators = -1
        self.percent_cooperators = 0
        self.tag_counter = Counter()
        self.max_size: int = max_size
        self.trustworthiness: float = 0

    def add_agent(self, agent: MigrationAgent):
        """Add a new member to the group. The member must be of type Agent."""
        self.members.add(agent)
        agent.group = self

    def is_accepting_new_members(self) -> bool:
        """Returns True if the group is is accepting new members, False otherwise."""

        return len(self.members) < self.max_size

    def step(self):
        """Update group stats."""

        self.num_cooperators = [x.status.move for x in self.members].count(True)
        self.percent_cooperators = (self.num_cooperators * 100) / len(self.members)

        group_size = len(self.members)
        migrate = self.model.config.migrate
        payoff_calculator = self.model.payoff_calculator

        min_payoff, max_payoff = self.model.payoff_range_calculator(len(self.members))

        # there are only two payoffs: one for all cooperators and one for all defectors
        if self.num_cooperators > 0:
            payoff_c = payoff_calculator(self.num_cooperators, group_size, True)

        if self.num_cooperators < group_size:
            payoff_d = payoff_calculator(self.num_cooperators, group_size, False)

        self.tag_counter = Counter([x.tag for x in self.members])

        # update members
        for agent in self.members:
            temp_payoff = payoff_c if agent.status.move else payoff_d

            # Normalize the payoff
            if migrate:
                agent.status.payoff = (temp_payoff - min_payoff) / (
                    max_payoff - min_payoff
                )
            else:
                agent.status.payoff = temp_payoff

            # update the trust values
            agent.update_trust(self.tag_counter)

    def __iter__(self):
        """Iterate through the member agents in the group."""
        return self.members.__iter__()

    def __len__(self):
        """Return the number of members in the group."""
        return len(self.members)


def migrate_agent(agent: MigrationAgent, source: Group, destination: Group):
    """Migrate an agent from one group to another

    Parameters
    ----------
    agent : MigrationAgent
        the migrating agent
    source : Group
        the group the agent will migrate away from
    destination : Group
        the group the agent will migrate to
    """

    source.members.remove(agent)
    destination.members.add(agent)

    agent.group = destination
    agent.status.update_stats_for_migration()

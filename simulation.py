"""Simulation Module.

Run the simulation.
"""

import argparse
import random
from multiprocessing import Process, current_process
from pathlib import Path
from typing import Union
from random import seed, random, choice
from math import exp
from enum import Enum
from functools import partial
from dataclasses import fields
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import networkx as nx
from model import MigrationModel, MigrationAgent, Group, ModelConfig
from games import Game, GameType

BETA = 0.2
RUNS_FILE = "runs.txt"
SEEDS_FILE = "seeds.txt"

# use the same seed for all simulations
SEED = 48092


def main(args):
    """Main."""
    runs = args.pop("runs")
    run_hash = args.pop("run_id")

    output_dir = Path(args["output_dir"])
    with open(output_dir / RUNS_FILE, "a", encoding="utf-8") as output_file:
        output_file.write(f"{run_hash}: {runs}\n")

    with open(output_dir / SEEDS_FILE, "a", encoding="utf-8") as output_file:
        output_file.write(f"{run_hash}: {SEED}\n")

    assert Path(args["graph_path"]).exists(), "graph doesn't exist"

    run_name = output_dir / f"{run_hash}"
    for i in range(runs):
        process_name = f"{run_hash}_{i}"
        run_process = Process(target=run_model, name=process_name, kwargs=args)
        run_process.start()


def parse_arguments():
    """Parse arguments passed from the commandline."""
    parser = argparse.ArgumentParser(
        description="Run a MigrationModel simulation.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--num_agents",
        help="the number of agents",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_groups",
        help="the number of groups",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_tags",
        help="the number of tags in the population",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--game_type",
        help="either nipd or nisd",
        required=True,
    )
    parser.add_argument(
        "--cost", help="the cost of cooperation [0, 1]", type=float, required=True
    )
    parser.add_argument(
        "--cooperation_threshold",
        help="the min level of cooperation required for generating benefits [0, 1]",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--trust_threshold",
        help="the min level of trust required by an agent so it does not seek migration [0, 1]",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--mutation_prob",
        help="the probability of move mutation [0, 1]",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--timesteps",
        help="the number of simulation timesteps",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--runs",
        help="the number of monte carlo runs",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_group_ratio",
        help="the max group size in terms of population size [0, 1]",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--data_to_save",
        help="how much data to save [0, 1]",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--migration_wait_period",
        help="how many timesteps to wait before migrating",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--decay",
        help="the rate at which trust decays [0,1]",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--run_id",
        help="the id of the job run",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--migration_strategy",
        help="method used by agents to select migration destination (no_migration, random, proportional_trust, most_trusted_group)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--graph_path",
        help="the path for the graph if social networking is enabled",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="the directory where the results will be written",
        required=True,
    )

    args = parser.parse_args()
    return args


def save_tag_data(datacollector, timesteps_to_save, tags, groups, fname, idx):
    """Save the data about the tags in the model."""

    # multi-index
    index_tuples = [(*idx[0], a, b) for a in tags for b in timesteps_to_save]
    index = pd.MultiIndex.from_tuples(index_tuples, names=idx[1] + ["tag", "timestep"])

    # create multi-indexed dataframe for information about tags
    tag_df = datacollector.get_table_dataframe("tag")
    tag_info = np.array([y[1:] for x in tags for y in tag_df[x]])
    columns = ["members", "cooperators", "migrators", "trustworthiness"]
    tag_df = pd.DataFrame(tag_info, index=index, columns=columns)
    save_as_parquet_table(tag_df, fname, "tag")

    # another dataframe for tag-group composition
    tag_group_df = datacollector.get_table_dataframe("tags_in_groups")
    tag_group_info = np.array([y for x in tags for y in tag_group_df[x]])
    tag_group_df = pd.DataFrame(tag_group_info, index=index, columns=groups)
    save_as_parquet_table(tag_group_df, fname, "tags_in_groups")


def save_data(datacollector, timesteps_to_save, num_groups, fname, idx):
    """Save the data about the model's stats."""

    # save the model data as one dataframe
    model_df = datacollector.get_model_vars_dataframe()
    model_df.index = pd.MultiIndex.from_tuples(((*idx[0], ts) for ts in timesteps_to_save), names=idx[1]+["timestep"])
    save_as_parquet_table(model_df, fname, "model")

    # multi-index
    index_tuples = [(*idx[0], a, b) for a in range(num_groups) for b in timesteps_to_save]
    index = pd.MultiIndex.from_tuples(index_tuples, names=idx[1] + ["group", "timestep"])

    # group information
    group_df = datacollector.get_table_dataframe("group")
    group_info = np.array([y for x in range(num_groups) for y in group_df[x]])
    columns = ["size", "cooperators", "trustworthiness"]

    # save the group info as one dataframe
    group_df = pd.DataFrame(group_info, index=index, columns=columns)
    save_as_parquet_table(group_df, fname, "group")


def save_agent_info(agents, fname, idx):
    """Save information about the agents to a datafame."""
    agent_df = pd.DataFrame([x.as_dict() for x in agents])
    agent_df = agent_df.set_index('id')
    agent_df.index = pd.MultiIndex.from_tuples(((*idx[0], i) for i in agent_df.index), names=idx[1]+['id'])
    save_as_parquet_table(agent_df, fname, "agents")


def save_as_parquet_table(dataframe, prefix, tag):
    """Save the specified pandas dataframe as a parquet table."""
    pq.write_table(pa.Table.from_pandas(dataframe), f"{prefix}_{tag}.parquet")


def setup_model(config):
    """Setup the model using the passed parameters"""

    run_seed = sum(int(x) for x in current_process().name.split("_"))
    seed(run_seed)
    np.random.seed(run_seed)

    # create model config object
    keys = (f.name for f in fields(ModelConfig) if f.name in config)
    config_dict = {k: config[k] for k in keys}
    config_dict["max_group_size"] = config["num_agents"] * config["max_group_ratio"]
    model_config = ModelConfig(**config_dict)

    # read graph for creating the social network of agents
    graph = nx.read_gpickle(config["graph_path"])

    # payoff_calculator for the model
    game = Game(GameType(config["game_type"].lower()))
    payoff_calculator = lambda n, N, m: game.calculate_payoff(
        config["cost"], n, N, m, config["cooperation_threshold"]
    )
    payoff_range_calculator = partial(
        game.payoff_range, config["cost"], config["cooperation_threshold"]
    )

    # method for updating trust
    trust_updater = partial(update_trust_in_tag, decay=config["decay"])

    model = MigrationModel(
        model_config, graph, payoff_calculator, payoff_range_calculator, trust_updater
    )

    # do some setup for each agent in the model
    migration_strategy = MigrationStrategy(MigrationMethod(config['migration_strategy']))
    for agent in model.schedule.agents:
        # [MIGRATION STRATEGY]
        # we assign the same migration strategy to each agent.
        agent.migration_strategy = migration_strategy

        # [TRUST IN TAGS]
        # random trust values
        agent.status.trust_in_tags = {t: random() for t in model.tags}
    return model


def run_model(**kwargs):
    """Run the model with the given inputs and write output."""

    model = setup_model(kwargs)

    timesteps = kwargs["timesteps"]
    process_name = current_process().name

    # collect data for only the required iterations
    data_collection_iteration = int(timesteps * (1 - kwargs["data_to_save"]))
    for i in range(timesteps):
        model.step()
        if i >= data_collection_iteration:
            model.collect_data()
    timesteps_to_save = range(data_collection_iteration + 1, kwargs["timesteps"] + 1)

    # save the model data as one dataframe
    output_dir = Path(kwargs["output_dir"])
    idx_keys = ['migration_strategy', 'cost', 'num_groups', 'num_tags', 'trust_threshold', 'migration_wait_period']
    idx_values = tuple(kwargs[k] for k in idx_keys)
    idx_names = ['ms', 'c', 'ng', 'nt', 'th', 'mwp']
    save_data(
        datacollector=model.datacollector,
        timesteps_to_save=timesteps_to_save,
        num_groups=kwargs["num_groups"],
        fname=output_dir / process_name,
        idx=(idx_values, idx_names)
    )
    save_tag_data(
        datacollector=model.datacollector,
        timesteps_to_save=timesteps_to_save,
        tags=model.tags.keys(),
        groups=[g.unique_id for g in model.groups],
        fname=output_dir / process_name,
        idx=(idx_values, idx_names)
    )

    ## save final status of agents
    save_agent_info(model.schedule.agents, output_dir / process_name, idx=(idx_values, idx_names))

    print("done", process_name)


def update_trust_in_tag(
    prev_trust: float, payoff: float, tag_size: int, total_neighbours: int, decay: float
) -> float:
    """Update the trust in a tag

    Parameters
    ----------
    prev_trust : float
        the previous trust in the tag
    payoff : float
        the agent's payoff in the current timestep
    tag_size : int
        the number of tags in the current group
    total_neighbours : int
        the total number of neighbours
    decay : float
        value that controls how the previous trust decays

    Returns
    -------
    float
        Updated trust in the tag
    """

    return (1 - decay) * prev_trust + (decay * payoff * tag_size) / total_neighbours


class MigrationMethod(Enum):
    """The migration method used by the agents."""

    # No Migration
    NO_MIGRATION = "no_migration"

    # Random Migration
    RANDOM = "random"

    # Payoff based migration
    PAYOFF = "payoff_based"

    # Based on trust in the group
    PROPORTIONAL_GROUP_TRUST = "proportional_group_trust"

    # Based on trust in a social connection
    MOST_TRUSTED_AGENT = "most_trusted_agent"

    # most trusted group
    MOST_TRUSTED_GROUP = "most_trusted_group"


# pylint: disable=too-few-public-methods
class MigrationStrategy:
    """catalog of different migration strategies"""

    def __init__(self, method: MigrationMethod):
        self.method = method

        self._destination_selectors = {
            MigrationMethod.RANDOM: self._random,
            MigrationMethod.PROPORTIONAL_GROUP_TRUST: self._proportional_group_trust,
            MigrationMethod.MOST_TRUSTED_GROUP: self._most_trusted_group,
        }

    def get_migration_destination(
        self,
        agent: MigrationAgent,
    ) -> Union[Group, None]:
        """Choose a migration destination using the supplied migration method for an agent.

        Parameters
        ----------
        agent : MigrationAgent
            the agent for which to select the migration destination
        options : set[Group]
            the options that are available for migration
        method : MigrationMethod
            the method to use for choosing the migration destination

        Returns
        -------
        Group
            the group that this agent should migrate to. None if it should not migrate to any
            of the available options.
        """

        return self._destination_selectors[self.method](agent)

    @staticmethod
    def _no_migration(agent: MigrationAgent) -> Union[Group, None]:
        """Do not migrate. So return None"""

        return None

    @staticmethod
    def _random(agent: MigrationAgent) -> Union[Group, None]:
        """Choose a random migration destination.

        Parameters
        ----------
        agent : MigrationAgent
            the agent for which to select the migration destination
        options : set[Group]
            the options that are available for migration

        Returns
        -------
        Group
            the group that this agent should migrate to. None if it should not migrate to any
            of the available options.
        """

        options = agent.get_migration_options()
        if len(options) == 0:
            return False

        # select a random group from the list
        dest = choice(options)

        # check whether the group is accepting new members
        if not dest.is_accepting_new_members():
            return None

        return dest

    @staticmethod
    def _proportional_group_trust(agent: MigrationAgent) -> Union[Group, None]:
        """Select a migration destination based on an agent's trust in an entire group.

        Parameters
        ----------
        agent : MigrationAgent
            the agent for which to select the migration destination

        Returns
        -------
        Group
            the group that this agent should migrate to. None if it should not migrate to any
            of the available options.
        """

        options = agent.get_migration_options()
        if len(options) == 0:
            return False

        # select a random group from the list
        dest = choice(options)

        # check whether the group is accepting new members
        if not dest.is_accepting_new_members():
            return None

        # compare the trust in the agent's current group vs. trust in the destination group
        fermi_prob = 1.0 / (
            1
            + exp(
                -1 * BETA * (agent.trust_in_group(dest) - agent.status.trust_in_group)
            )
        )
        return dest if random() < fermi_prob else None

    @staticmethod
    def _most_trusted_group(agent: MigrationAgent) -> Union[Group, None]:
        options = agent.get_migration_options()
        if len(options) == 0:
            return None

        most_trusted = max(options, key=lambda g: agent.trust_in_group(g))

        # check whether the group is accepting new members
        if not most_trusted.is_accepting_new_members():
            return None

        return most_trusted


if __name__ == "__main__":
    # get the arguments passed as parameters
    run_args = vars(parse_arguments())

    # update the config with these parameter values
    # config.update((k, run_args[k]) for k in config.keys() & run_args.keys())

    main(run_args)

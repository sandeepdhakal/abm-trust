"""Module to submit jobs to HPC."""

import subprocess
from itertools import product
from pathlib import Path

DIR = 'output'

params = {
    "num_agents": 1000,
    "game_type": "NISD",
    "mutation_prob": 0.01,
    "timesteps": 5000,
    "runs": 50,
    "data_to_save": 1,
    "max_group_ratio": 0.25,
    "decay": 0.05,
    "cooperation_threshold": 0.5,
    "output_dir": f"{DIR}",
    "graph_path": "networks/sn_d8.graph",
    "cost": 0.1,
    "num_groups": 10,
    "num_tags": 2,
    "trust_threshold": 0.5,
    "migration_wait_period": 0,
    "migration_strategy": "random",
}

migration_strategies = {
    'no_migration': 0,
    'random': 1,
    'payoff_based': 2,
    'proportional_group_trust': 3,
    'most_trusted_agent': 4,
    'most_trusted_group': 5
}


def get_qsub_command(params_fname, log_dir, run_id):
    """Return the qsub command for the passed parameters."""

    error_file = f"{log_dir}/{run_id}.error"
    output_file = f"{log_dir}/{run_id}.output"
    params_string = f"FILE=@{params_fname}"
    cmd = f"qsub -e {error_file} -o {output_file} -v {params_string} run-simulation.sh"
    return cmd


if __name__ == "__main__":
    costs = [x / 10 for x in range(0, 10)]
    num_groups = [10]
    num_tags = [2,3,5]
    migration_wait_periods = [0]
    trust_thresholds = [0.5]
    mms = ['most_trusted_group']

    num_jobs = 0
    num_jobs_failed = 0

    param_combinations = list(
        product(costs, num_groups, num_tags, migration_wait_periods, trust_thresholds, mms)
    )
    for c, ng, nt, mwp, th, mm in param_combinations:
        run_id = hash((c, ng, nt, mwp, th, migration_strategies[mm])) % (2 ** 31 - 1)

        params["run_id"] = run_id
        print(run_id)

        params.update(
            {
                "cost": c,
                "num_groups": ng,
                "num_tags": nt,
                "migration_wait_period": mwp,
                "trust_threshold": th,
                "migration_strategy": mm,
            }
        )

        params_fname = f"{params['output_dir']}/{run_id}.params"
        path = Path(params_fname)
        assert not path.exists(), "file exists"

        with open(params_fname, "w", encoding="utf-8") as f:
            f.writelines((f"--{k}={v}\n" for k, v in params.items()))

        #qsub_command = get_qsub_command(params_fname, f"{DIR}/logs", run_id)
        #exit_status = subprocess.call(qsub_command, shell=True)
        #if exit_status == 1:
        #   print(f"Job {qsub_command} failed to submit")
        #   num_jobs_failed += 1
        #else:
        #   num_jobs += 1

#    print(f"{num_jobs} jobs submitted, {num_jobs_failed} jobs not submitted")

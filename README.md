# Agent-based model

Scripts to run an agent-based model simulation of cooperation and trust with tags for migration decisions. The results of this study have been published in the journal _Royal Society Open Science_.\
Dhakal S, Chiong R, Chica M, Han TA. 2022 Evolution of cooperation and trust in an N-player social dilemma game with tags for migration decisions. _R. Soc. Open Sci._ **9**: 212000. https://doi.org/10.1098/rsos.212000

## Installation

Requires [Mesa](https://github.com/projectmesa/mesa).
For all requirements, see ``requirements.txt``

```bash
pip install -r requirements.txt
```

## Usage
```bash
python simulation.py @config_file
```

See ``simulation.py`` for the format of the config files. To submit the simulation job on an HPC, see ``submit-jobs.py``.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

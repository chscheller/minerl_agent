# Sample Efficient Reinforcement Learning through Learning from Demonstrations in Minecraft

This repository contains the code of our third-placed submission to the [NeurIPS 2019: MineRL Competition](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition).

See our full paper for details: https://arxiv.org/abs/2003.06066.

## Run

Download the [MineRL dataset](https://minerl.io/dataset/) to ``./data``.

```shell script
# Run local training and testing
python run.py

# Run local training and testing (headless)
xvfb-run -a python run.py
```

## Authors

[Yanick Schraner](https://github.com/YanickSchraner/) and
[Christian Scheller](https://github.com/metataro/)

## Acknowledgement

The IMPALA implementation is based on https://github.com/deepmind/scalable_agent.

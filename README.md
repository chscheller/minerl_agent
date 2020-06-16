# Sample Efficient Reinforcement Learning through Learning from Demonstrations in Minecraft

This repository contains the code of our 3rd placed submission to the [NeurIPS 2019: MineRL Competition](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition).

See our full paper for details: https://arxiv.org/abs/2003.06066.

The IMPALA implementation is build upon https://github.com/deepmind/scalable_agent.

# Install

```shell script
git clone https://github.com/metataro/minerl_agent.git 

cd minerl_agent

pip install -r requirements.txt
```

# Run

Download the [MineRL dataset](https://minerl.io/dataset/) to ``./data``.

```shell script
# Run local training and testing
python run.py

# Run local training and testing (headless)
xvfb-run -a python run.py
```
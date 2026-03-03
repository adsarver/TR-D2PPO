# Racing Reinforcement Learning
Setup by cloning `https://github.com/f1tenth/f1tenth_gym`
Install using `pip install -e .`

# Methodology

D2PPO (DPPO with Dispersive loss) and DDIM
Run `python weight_initializer.py --load demos/expert_demos.pt` for pretraining with pregenerated demos, otherwise `python weight_initializer.py`
# Space DQN

## Environment Introduction

This is a game project based on reinforcement learning. It uses the Deep Q Network (DQN) algorithm to train agents to survive and score in the game. The goal of the game is to control the player to avoid meteorites and shoot meteorites to get high scores.

## Game Rules and Goals

- Players can move up, down, left, and right, and can shoot.

- Meteorites will be randomly generated from the top of the screen and move downward.

- Players need to avoid collisions with meteorites and can shoot meteorites to get points.

- The goal of the game is to survive as long as possible and get high scores.

- The winning condition of the agent is to get a 200-point reward

## State Space

The state space contains various information about the player and the environment, including the player's position, speed, direction, health, laser cooling time, meteorite and laser information. The details of the state space are as follows:

| State variables | Description |
|---------------------|------------------------------------|
| player_x | Player x coordinate |
| player_y | Player y coordinate |
| player_vx | Player x direction speed |
| player_vy | Player y direction speed |
| player_direction_x | Player x direction |
| player_direction_y | Player y direction |
| lives | Player health |
| can_shoot | Whether the player can shoot |
| is_invulnerable | Whether the player is invulnerable |
| laser_cooldown | Laser cooling time |
| active_lasers | Current number of lasers |
| meteor_{i}_x | The x coordinate of the ith meteorite |
| meteor_{i}_y | The y coordinate of the ith meteorite |
| meteor_{i}_dist | The distance between the ith meteorite and the player |
| meteor_{i}_vx | The x direction speed of the ith meteorite |
| meteor_{i}_vy | y-speed of the ith meteorite |
| meteor_{i}_h_dist | x-distance of the ith meteorite from the player |
| meteor_{i}_v_dist | y-distance of the ith meteorite from the player |
| laser_{i}_x | x-coordinate of the ith laser |
| laser_{i}_y | y-coordinate of the ith laser |
| laser_{i}_dist | distance of the ith laser from the player |

## Action Space

The action space contains all actions that the player can perform, including moving and shooting. The action space details are as follows:

| Action | Description |
|---------------------|------------------------------------|
| NONE | No action |
| LEFT | Move left |
| RIGHT | Move right |
| UP | Move up |
| DOWN | Move down |
| UP_LEFT | Move up left |
| UP_RIGHT | Move up right |
| DOWN_LEFT | Move down left |
| DOWN_RIGHT | Move down right |
| SHOOT | Shoot || SHOOT_LEFT | Shoot left |
| SHOOT_RIGHT | Shoot right |
| SHOOT_UP | Shoot up |
| SHOOT_DOWN | Shoot down |
| SHOOT_UP_LEFT | Shoot up left |
| SHOOT_UP_RIGHT | Shoot up right |
| SHOOT_DOWN_LEFT | Shoot down left |
| SHOOT_DOWN_RIGHT | Shoot down right |

## Rewards and Penalties

| Rewards/Penalties | Description |
|---------------------|------------------------------------|
| SURVIVAL_REWARD | Survival bonus per frame |
| MOVEMENT_REWARD | Movement bonus |
| SHOOT_REWARD | Shooting bonus |
| NO_MOVEMENT_PENALTY | No movement penalty |
| METEOR_HIT_PENALTY | Meteor hit penalty |
| METEOR_DESTROY_REWARD | Meteor destroyed reward |
| TARGET_REWARD | Target hit reward |

## Environment setup

- Please make sure to install the following dependencies:

```sh
pip install torch numpy pygame matplotlib ipython
```

## Start the game
- To start the game, run the following command:

```sh
python code/main.py
```

## Start training

- To start training the agent, run the following command:

```sh
python code/train.py --episodes <number_of_episodes> [--render]
```

- Optional parameters:

- --episodes: The number of episodes to train, default is 1000.
- --render: Enable rendering and visual effects (slower).

## Play

- To play with the trained model, run the following command:

```sh
python code/play.py --model <path_to_trained_model> [--episodes <number_of_episodes>]
```

- Optional parameters:

- --model: The path to the trained model file.
- --episodes: The number of episodes to play with, default is 10.
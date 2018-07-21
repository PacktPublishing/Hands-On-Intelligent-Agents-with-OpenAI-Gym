# Chapter 6: Implementing an Intelligent Agent for Optimal Control using Deep Q Learning

1. [Brief chapter summary and outline of topics covered]()
2. [Code structure]([#code-structure])
3. [Running the code](#running-the-code)
   - [Training](#training)
   - [Testing](#testing)

## 2. Code structure
## 3. Running the code

The [deep_Q_learner.py](./deep_Q_learner.py) is the main script that takes care of training and testing depending on the script's arguments. The table below summarizes the arguments that the script supports and what they mean. Note that, most of the agent and environment related configuration parameters are in the [parameters.json](parameters.json) file and only the few parameters that are more useful when launching the training/testing scripts are made available through the command line arguments.

| Argument                 | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `--params-file`          |                                                              |
| `--env`                  | Name/ID of the Atari environment available in OpenAI Gym. Default=`SeaquestNoFrameskip-v4` |
| `--gpu-id`               | ID of the GPU device to be used. Default=`0`                 |
| `--render`               | Render the environment to screen. Off by default             |
| `--test`                 | Run the script in test mode. Learning will be disabled. Off by default |
| `--record`               | Enable recording (Video & stats) of the agent's performance  |
| `--recording-output-dir` | Directory to store the recordings (video & stats). Default=`./trained_models/results` |




- ### Training

- ### Testing

  `python deep_Q_learner.py --env RiverraidNoFrameskip-v4 --test --render  --record`
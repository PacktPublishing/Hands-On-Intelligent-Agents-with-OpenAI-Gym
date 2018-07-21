#  Setup instructions for Hands-on Intelligent Agents with OpenAI Gym

This is the code repository for the [Hands-On Intelligent Agents with OpenAI Gym](https://www.packtpub.com/big-data-and-business-intelligence/hands-intelligent-agents-openai-gym) book.

To setup the necessary packages to run the code samples for the chapters effortlessly, a conda environment description file is provided. So, the only/main system requirement is the Anaconda python distribution. You can manually install the Anaconda python distribution for your OS/platform from the [official Anaconda website's download page](https://www.anaconda.com/download/). Below, you will find scripts or commands to automate the installation process.

Follow the setup instructions based on your OS:

- [Mac-OSX](#mac-osx-one-click-setup-for-mac-osx-using-a-bash-script)

- [Ubuntu](#ubuntu-quick-setup-for-ubuntu)

  

### [Mac OSX] One-click setup for Mac OSX using a bash script

You can use the [macOS/setup.sh](./macOS/setup.sh) script to automate the process of installing OpenAI Gym, PyTorch and other required packages for the Agent development inside a (mini) conda environment on Mac OSX.  Launch the script from a terminal and you will see the installer's message like below: (It will be colorful on your computer)

```
                                ▄▄▄▄▄▄
                              ▄▓▀    ▓▓▓██▄▄
                           ▄▄▓▓  ▄▄▓▀▀     ▀▓▄
                         ▄▓▀ ▓▌ ▐▓  ▄▄█▀▓▓▄  ▓▌
                        ▐▓▌  ▓▌ ▐▓█▀▀▀▄▄  ▀▀▓▓▌
                         ▓▌  ▓▌ ▐▌    ▐▓▀█▄  ▀▓▄
                          ▓▓▄ ▀▀▀▓▄  ▄▄▓  ▓▌  ▓▌
                          ▓▌▀▀▓▄▄▄▄▓▀ ▐▓  ▓▌ ▄▓▀
                          ▀▓▄   ▀▀  ▄▄▓▀ ▐▓▓▓▀
                            ▀▓▄▄▄▄▓▀▀    ▓▓
                                  ▀▓▓▓▓▓▀▀
  ▄▄▄▓▓▄▄▄                                          ▄▄▄      ████████
▄▓▓▀▀  ▀▀▓▓▄                                       ▓▓▀▓▌        ▓▓▌
▓▓▌      ▐▓▓  ▓▓▄▓▓▓▓▓▄   ▄▓▓▀█▓▄   ▓▓▄▓▓▓▓▓▄     ▓▀  ▓▓▄       ▓▓▌
▓▓▌      ▐▓▓  ▓▓    ▀▓▓  ▓▓▌    ▓▓  ▓▓▌   ▐▓▓    ▐▓▌   ▓▓       ▓▓▌
▓▓▌      ▐▓▓  ▓▌    ▐▓▓  ▓▓▓██████  ▓▓▌   ▐▓▓   ▐▓▓▄▄▄▄▓▓▓      ▓▓▌
▀▓▓▄    ▄▓▓▀  ▓▓▄   ▄▓▓  ▓▓▌    ▄▄  ▓▓▌   ▐▓▓   ▓▓▀▀▀▀▀▀▓▓▌     ▓▓▌
  ▀▀█▓▓█▀▀    ▓▌▀████▀    ▀▀█▓██▀   ▓▓▌   ▐▓▓  ██▀       ██  ████████
              ▓▌                                                     
              ▓▌           
              
 Setting up Gym & dependencies. Takes 5-15 minutes, based on Internet speed.
```

Followed by a series of step. If all goes well, you should get a message which looks like below:

```console
  ███████╗ ██╗   ██╗  ██████╗  ██████╗ ███████╗ ███████╗ ███████╗    ██╗
  ██╔════╝ ██║   ██║ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝    ██║
  ███████╗ ██║   ██║ ██║      ██║      █████╗   ███████╗ ███████╗    ██║
  ╚════██║ ██║   ██║ ██║      ██║      ██╔══╝   ╚════██║ ╚════██║    ╚═╝
  ███████║ ╚██████╔╝ ╚██████╗ ╚██████╗ ███████╗ ███████║ ███████║    ██╗
  ╚══════╝  ╚═════╝   ╚═════╝  ╚═════╝ ╚══════╝ ╚══════╝ ╚══════╝    ╚═╝

HOIAWOG setup complete.
Use 'import gym' to use Gym in python files

To rerun the example agent, enter these commands in terminal:
    source ~/.bash_profile
    source activate rl_gym_book
    cd 
    python example_agent.py

For next steps, check out the README file at 
https://github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym

```
### [Ubuntu] Quick setup for Ubuntu

1. You can install `Anaconda3-4.3.0` using the following bash commands:

   `wget http://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh -O ~/anaconda.sh`

   `bash ~/anaconda.sh -b -f -p $HOME/anaconda`

2. Then create a conda environment named `rl_gym_book` with all the required packages as mentioned in [conda_env.yml](../conda_env.yml) using the following command:

   `conda env create -f ../conda_env.yaml -n rl_gym_book python=3.5`

3. Activate the `rl_gym_book` conda environment before running any code:

   `source activate rl_gym_book`
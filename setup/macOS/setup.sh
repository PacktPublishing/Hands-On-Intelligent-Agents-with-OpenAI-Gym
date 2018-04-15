#!/bin/sh
set -e

command_exists () {
    type "$1" &> /dev/null ;
}

safe_brew_install () {
    while [ $# -gt 0 ]
    do
        if brew ls --versions "$1" > /dev/null ; then 
           echo "[CHECK] $1 already installed."
        else
           echo "Installing $1..."
           brew install "$1"
        fi
    shift
    done
}

tput setaf 105
echo "                                    ▄▄▄▄▄▄"
echo "                                  ▄▓▀    ▓▓▓██▄▄"
echo "                               ▄▄▓▓  ▄▄▓▀▀     ▀▓▄"
echo "                             ▄▓▀ ▓▌ ▐▓  ▄▄█▀▓▓▄  ▓▌"
echo "                            ▐▓▌  ▓▌ ▐▓█▀▀▀▄▄  ▀▀▓▓▌"
echo "                             ▓▌  ▓▌ ▐▌    ▐▓▀█▄  ▀▓▄"
echo "                              ▓▓▄ ▀▀▀▓▄  ▄▄▓  ▓▌  ▓▌"
echo "                              ▓▌▀▀▓▄▄▄▄▓▀ ▐▓  ▓▌ ▄▓▀"
echo "                              ▀▓▄   ▀▀  ▄▄▓▀ ▐▓▓▓▀"
echo "                                ▀▓▄▄▄▄▓▀▀    ▓▓"
echo "                                      ▀▓▓▓▓▓▀▀"
echo "      ▄▄▄▓▓▄▄▄                                          ▄▄▄      ████████"
echo "    ▄▓▓▀▀  ▀▀▓▓▄                                       ▓▓▀▓▌        ▓▓▌"
echo "    ▓▓▌      ▐▓▓  ▓▓▄▓▓▓▓▓▄   ▄▓▓▀█▓▄   ▓▓▄▓▓▓▓▓▄     ▓▀  ▓▓▄       ▓▓▌"
echo "    ▓▓▌      ▐▓▓  ▓▓    ▀▓▓  ▓▓▌    ▓▓  ▓▓▌   ▐▓▓    ▐▓▌   ▓▓       ▓▓▌"
echo "    ▓▓▌      ▐▓▓  ▓▌    ▐▓▓  ▓▓▓██████  ▓▓▌   ▐▓▓   ▐▓▓▄▄▄▄▓▓▓      ▓▓▌"
echo "    ▀▓▓▄    ▄▓▓▀  ▓▓▄   ▄▓▓  ▓▓▌    ▄▄  ▓▓▌   ▐▓▓   ▓▓▀▀▀▀▀▀▓▓▌     ▓▓▌"
echo "      ▀▀█▓▓█▀▀    ▓▌▀████▀    ▀▀█▓██▀   ▓▓▌   ▐▓▓  ██▀       ██  ████████"
echo "                  ▓▌                                                     "
echo "                  ▓▌                                                     "
tput sgr0
echo; echo "Setting up Gym & dependencies. Takes 5-15 minutes, based on internet speed."

read -rsp $'>> Press enter to begin <<\n'

echo; echo "**** OPENAI GYM SETUP SCRIPT ****"
echo "[PART 1] Setup Homebrew & system dependencies"
echo "*********************************"; sleep 1; echo
echo "Reaching out to Homebrew..."

if command_exists brew ; then
    echo "[CHECK] Homebrew already installed. Updating Homebrew."
    brew update
    echo "[CHECK] Homebrew successfully updated."
else
    echo "Installing Homebrew. Enter your system password at prompt, then press enter."
    tput smul
    echo "[TIP] After entering password, it takes awhile."
    tput rmul
    sleep 8
    echo "Downloading Homebrew..."
    # Install Homebrew
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    echo; echo "[CHECK] Successfully installed Homebrew."    
fi

touch ~/.bash_profile
echo; echo "Install Xcode Command Line Tools..."
echo "If you have already installed Xcode CLT, you will see an error. That is fine."
read -rsp $'>> Press enter to continue <<\n'

set +e
xcode-select --install
set -e
if command_exists xcode-select ; then 
    echo; echo "[CHECK] Xcode Command Line Tools successfully configured."
else
    echo "Failed to install Xcode Command Line tools. Exiting"
    exit 0
fi

safe_brew_install cmake swig boost boost-python sdl2 wget

echo; echo "[PART 1] Success!"
echo; echo "**** OPENAI GYM SETUP SCRIPT ****"
echo "[PART 2] Setup Python 3.5 / Conda"
tput smul
echo "[TIP] Say 'yes' to each prompt that asks"
echo "[TIP] Scroll down the license by holding enter"
tput rmul
echo "*********************************"; echo
read -rsp $'>> Press enter to continue <<\n'


source ~/.bash_profile
if command_exists conda ; then
    echo "[CHECK] Conda already installed."
    echo "Updating conda..."
    conda update conda
    case "$(python --version 2>&1)" in
    *" 3.5"*)
        echo "Using Python 3.5 already. Continuing..."
        ;;
    *)
        echo "Switching to Python 3.5 using Conda..."
        
        set +e
        conda create -n p35 python=3.5
        set -e
        source activate p35
        tput smul
        echo "[TIP] New terminal tabs/windows must run 'source activate p35' for Gym"
        echo "[TIP] Add the above command to your .bash_profile for auto-activation"
        tput rmul
        read -rsp $'>> Press enter to continue <<\n'
        ;;
     esac
else
    # Install conda
    echo "Installing Miniconda python package/environment manager..."
    safe_brew_install wget
    
    wget -c -nc https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    chmod +x Miniconda3-latest-MacOSX-x86_64.sh
    ./Miniconda3-latest-MacOSX-x86_64.sh
    # Dying around here for some reason, suddenly. Is it -nc flag? Or...?
    echo "Finished installing Miniconda"
    rm Miniconda3-latest-MacOSX-x86_64.sh
    source ~/.bash_profile
    tput smul
    echo "[TIP] For Conda to work, type 'source ~/.bash_profile' after the script completes."
    tput rmul
    read -rsp $'>> Press enter to continue <<\n'
fi

echo; echo "[PART 2] Success!"
echo; echo "**** OPENAI GYM SETUP SCRIPT ****"
echo "[PART 3] Install OpenAI Gym"
tput smul
echo "[TIP] The pachi-py step takes awhile."
tput rmul
echo "*********************************"; sleep 1; echo

pip install 'gym[all]'

echo; echo "[PART 3] Success!"
echo; echo "**** OPENAI GYM SETUP SCRIPT ****"
echo "[PART 4] Download and run an example agent"
echo "*********************************"; sleep 1; echo

sleep 1
dir="GymAgents"
if [ "${PWD##*/}" != $dir ]; then 
     if [ ! -d $dir ]; then
          mkdir $dir
     fi
     cd "$dir"
fi


python setup_test.py

echo "[CHECK] Gym is working."
echo "[PART 4] Success!"

echo
tput setaf 118 
echo "  ███████╗ ██╗   ██╗  ██████╗  ██████╗ ███████╗ ███████╗ ███████╗    ██╗"
echo "  ██╔════╝ ██║   ██║ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝    ██║"
echo "  ███████╗ ██║   ██║ ██║      ██║      █████╗   ███████╗ ███████╗    ██║"
echo "  ╚════██║ ██║   ██║ ██║      ██║      ██╔══╝   ╚════██║ ╚════██║    ╚═╝"
echo "  ███████║ ╚██████╔╝ ╚██████╗ ╚██████╗ ███████╗ ███████║ ███████║    ██╗"
echo "  ╚══════╝  ╚═════╝   ╚═════╝  ╚═════╝ ╚══════╝ ╚══════╝ ╚══════╝    ╚═╝"
tput sgr0
echo
echo "OpenAI Gym setup complete."
echo "Use 'import gym' to use Gym in python files"
echo
echo "To rerun the example agent, enter these commands in terminal:"
echo "    source ~/.bash_profile"
echo "    source activate p35"
echo "    cd ${dir}"
echo "    python example_agent.py"
echo
echo "For next steps, check out the Gym docs"
echo "https://gym.openai.com/docs"
echo

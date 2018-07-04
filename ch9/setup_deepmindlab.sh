#!/usr/bin/env bash
# A bash script to automatically install Roboschool and its dependencies (Bullet3 and other system dependencies) from
# source for use with OpenAI Gym environment. This script is part of the repo at
# https://github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym for the book chapter 9
# Author: Praveen Palanisamy

set -e

echo "$(tput setaf 2) Step 1/2: Installing system dependencies for Bazel:$(tput sgr0)"
echo "$(tput setaf 2) Step 1.a) Installing Java/JDK:$(tput sgr0)"
sudo apt-get update && sudo apt-get install -y openjdk-8-jdk software-properties-common git liblua5.1-0-dev zip libsdl2-dev python-dev python3-dev libffi-dev python-numpy build-essential gettext libgl1-mesa-glx libgl1-mesa-dri
# PPA is needed on Ubuntu 14.04 LTS
sudo add-apt-repository -y ppa:webupd8team/java; sudo apt-get update && \
	yes 'yes' | sudo apt-get install -y oracle-java8-installer -y

echo "$(tput setaf 2) Step 1.b) Adding Bazel distribution URI as package source:$(tput sgr0)"
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
wget -qO- https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

echo "$(tput setaf 2) Step 1.c) Installing and updating Bazel $(tput sgr0)"
sudo apt-get update && sudo apt-get install -y bazel && sudo apt-get upgrade -y bazel

echo "$(tput setaf 2) Successfully installed and setup Bazel"
echo "$(tput setaf 2) Step 2/2 Cloning DeepMind Lab"
git clone https://github.com/deepmind/lab deepmindlab 
echo "$(tput setaf 2) Setup is now complete. You may now run a live example of a random agent by executing the following command"
echo "$(tput setaf 1) cd deepmindlab && bazel run :python_random_agent --define graphics=sdl -- --length=10000$(tput sgr0)"

#!/usr/bin/env bash
# A bash script to automatically install Roboschool and its dependencies (Bullet3 and other system dependencies) from
# source for use with OpenAI Gym environment. This script is part of the repo at
# https://github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym for the book chapter 9

set -e

echo "Step 1/4: Installing system dependencies for Roboschool & Bullet3"
sudo apt install -y cmake ffmpeg pkg-config qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.5-dev \
    libboost-python-dev libtinyxml-dev git

echo "Step 2/4: Cloning Roboschool code from git repo:"
mkdir -p "${HOME}/software"
cd "${HOME}/software"
ROBOSCHOOL_URL="https://github.com/openai/roboschool.git"
if [ ! -d "roboschool" ] ; then
	git clone $ROBOSCHOOL_URL
else
	cd "roboschool" && git pull $ROBOSCHOOL_URL && cd ..
fi
ROBOSCHOOL_PATH="${HOME}/software/roboschool"

echo "Step 3/4: Cloning & setting up Bullet3 code from git repo:"
BULLET3_URL="https://github.com/olegklimov/bullet3 -b roboschool_self_collision"
if [ ! -d "bullet3" ] ; then
	git clone $BULLET3_URL
else
	cd "bullet3" && git pull origin roboschool_self_collision && cd ..
fi
mkdir -p bullet3/build
cd bullet3/build
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=${ROBOSCHOOL_PATH}/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
make -j8 && make install
cd ../..

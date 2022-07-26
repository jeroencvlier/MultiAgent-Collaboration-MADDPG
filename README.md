# <u>Collaboration and Competition</u>
### Introduction

The goal of this project is to train 2 agents to play tennis against each other using either collaboration.


![GIF of Trained Network](Images/Tennis.gif)


### Environment

The environment consists of 2 rackets that hit a ball at each other. If the ball goes over the net, a reward of +0.1 is received. If the ball bounces on the ground or falls out of bounds, a reward of -0.01 is received. The goal is therefore to keep the ball in play. There are 2 continuous action spaces that the agent can play, and they relate to the movements left/right and up/down both in the range [-1:1]. The state space consists of 24 observations describing the position and velocity of the ball and racket. This is used to feed into the neural network.

To solve the environment, an average of +0.5 points needs to be achieved over 100 episodes. The score is taken from the max score of both agents.

### Getting Started

To run this project yourself, you will need to follow the installation instructions listed below:

1. Follow this [Udacity Github link](https://github.com/udacity/deep-reinforcement-learning#dependencies) for instructions to install all the dependencies.

2. Download [this repository](https://github.com/jeroencvlier/MultiAgent-Tennis-MADDPG) to your local computer.

3. Download and extract the environment inside the "MultiAgent-Tennis-MADDPG" folder you just downloaded. Use one of the following links. You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
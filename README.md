# schiphol-gym
This repository contains some experiments related to building environments to simulate certain processes around Schiphol Airport.


# gate-reassignment
The gate reassignment problem is something that needs to be solved at airports every day. Usually the day of operations begins with an existing gate schedule. However, during the day delays occur, flights get cancelled, flights are diverted etc. The gate-reassignment environment mimics this problem. It starts with a random gate schedule for a day and a flight that needs to be assigned a new gate because it was e.g. delayed and hence creates a conflict. The goal of this environment is to fit in this flight in the existing schedule with the minimal ammount of changes necessary.

The question you might ask here is why you would like to solve this using reinforcement learning. The motivation is actually quite simple: there is a lot of stochasticity involved in those gate schedules and the conflichts that occur (arrival and departure times of flights are uncertain). Morever, there is a potential for each change to cause a ripple effect in schedules later on.

# taxi-routing
This process involves airplanes that arrive on a runway and need to go to a gate or that need to go from a gate to runway to take off. No Gym environment for this is implemented yet, but there is a notebook showing some simple simulations that could be easilly turned into a gym environment.

# Disclaimer & License
This repository is purely based on open sources and knowledge. It is my own work and  unrelated to my affiliation to Schiphol Airport. 

The goal of this repository is to show how easily open sources can be used to create simulations of certain processes to be used in e.g. reinforcement learning. However, as you may have noticed this repository is provided without a Licence. This is done intentionally for now. If you want to use this code, contact me first.
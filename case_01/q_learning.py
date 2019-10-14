#!/usr/bin/python3

# Artificial Intelligence for Business
# Optimizing Warehouse Flows with Q-Learning

# Libraries
import numpy as np

# Setting the parameters gamma and alpha for the Q-Learning
gamma = 0.7
alpha = 0.9

# Part 1 - Defining the environment

#     Step 1 - Defining the States
location_to_state = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
}

#     Step 2 - Defining the Actions
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

#     Step 3 - Defining the Rewards
R = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
              ])

# Part 2 - Building the A.I solution with Q-Learning

#     Step 1 - Initializing the Q-Values
Q = np.array(np.zeros([12, 12]))

#     Step 2 - Implementing the Q-Learning process
for i in range(1000):
    current_state = np.random.randint(0, 12)
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions) 
    TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state] 
    Q[current_state, next_state] += alpha * TD

# Part 3 - Going to Production

# Making a mapping from the states to the locations
state_to_location = {state: location for location, state in location_to_state.items()}

# Making the final function that will return the optimal route
def route(starting_location, ending_location):
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

print(route('E', 'G'))
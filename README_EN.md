# Project Documentation

## [cite_start]Important: Required Libraries for Running [cite: 1]
[cite_start]In addition, recordings of rooms 1-3. [cite: 1]

## [cite_start]Installation and Running the Game: [cite: 2]

1.  [cite_start]**Extract the project folder to your computer.** [cite: 3]
2.  [cite_start]**Navigate to the folder via PowerShell.** [cite: 4]
3.  [cite_start]**Open a virtual environment to contain the exact environment.** [cite: 5]
    ```bash
    py -3.11 -m venv venv
    ```
4.  [cite_start]**Run the command that activates the virtual environment.** [cite: 6]
    ```bash
    ./venv/Scripts/Activate.ps1
    ```
5.  [cite_start]**You might need to run the following command beforehand.** [cite: 7]
    ```bash
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    ```
6.  [cite_start]**Install the required libraries.** [cite: 9]
    ```bash
    pip install -r requirements.txt
    ```
7.  [cite_start]**Run the main project file.** [cite: 10]
    ```bash
    python main.py
    ```
    [cite_start]This command will activate the rooms. [cite: 10]

## [cite_start]Usage Instructions and Screens: [cite: 11]

* [cite_start]When the game loads, the grid appears with the first room, and next to the grid, there's a console with messages about the actions taken in the game. [cite: 12]
* [cite_start]In each room, it's possible to refresh the room and get a new layout that is randomly initialized (unsolvable room iterations may occur). [cite: 13]
* [cite_start]Each room has the option to edit the settings of that space and the agent's learning method. [cite: 14]
* Each agent has the option to train "slowly" or "quickly." [cite_start]In slow mode, iterations can be seen in Room 1, and in Rooms 2 and 3, episodes can be observed occurring. [cite: 15]
* [cite_start]In Rooms 2-3, if slow training is chosen, there is also an option to skip to any episode within the training range (e.g., if training 5000 episodes, you can skip to episodes 2-4999). [cite: 16]
* [cite_start]The same episode can be run to observe the agent's learning state at that point. [cite: 17]
* [cite_start]Also, in all rooms, the final training can be run by clicking the `Run` button. [cite: 18]
* [cite_start]Animation can be paused with the `Pause` button, and the run can be reset with the `Reset Run` button. [cite: 19]
* [cite_start]In Rooms 2,3, you can click the `Show Q-Vals` button; this button is a toggle, and it will display the trained Q-table for that state on the grid. [cite: 20]
* [cite_start]In Rooms 2,3, you can click the `Plot Data` button, which will open a window (it's recommended to maximize it to full screen for comfortable viewing of the buttons) where you can see 3 different graphs: [cite: 21]
    * [cite_start]The first relates to rewards during training. [cite: 21]
    * [cite_start]The second relates to the number of steps during training. [cite: 21]
    * [cite_start]The third shows the grid and how many steps the agent took in each direction for each tile. [cite: 21]

## [cite_start]Room 1: Dynamic Programming [cite: 22]

[cite_start]The first agent uses Dynamic Programming, possessing a model of the environment and its rules. [cite: 23] [cite_start]It is not required to explore the environment but rather to calculate the ideal policy using the given model to meet the task according to the rules. [cite: 24] [cite_start]It starts with some random policy and then begins calculations to perform Policy Iteration. [cite: 25]

### [cite_start]The Challenge: [cite: 26]

[cite_start]The agent must navigate the grid to first collect the bag, then the rope, and finally reach the exit. [cite: 27] [cite_start]The main challenge is the presence of wall tiles and slippery tiles. [cite: 28] [cite_start]On slippery tiles, the action chosen by the agent is not significant, and the outcome is determined by a fixed probability (randomly rolled with each new map generation, where the minimum threshold for a legal direction, i.e., not towards a wall or outside the grid, is 20%, to prevent very low chances in certain directions). [cite: 28] [cite_start]The agent is forced to develop a policy that accounts for this uncertainty. [cite: 29]

[cite_start]Slippery tiles posed a problem in implementation in terms of how the direction appeared when the agent calculated the policy, because its decision has no meaning, and therefore there is no real meaning to the direction it "would most want" to move in. [cite: 30] [cite_start]Therefore, I added a mechanism for these tiles, where even though the choice is not important, the displayed policy will be such that when the policy converges, the arrows appearing on the tiles (i.e., where the agent should want to go) will point to the closest legal tile that has the highest expected reward value. [cite: 30]

[cite_start]Additionally, I limited the random placement of items relative to the center of the map. [cite: 31]

### [cite_start]Learning Method: Policy Iteration [cite: 32]

[cite_start]The algorithm repeats two steps until the policy stabilizes: [cite: 33]

1.  [cite_start]**Policy Evaluation** – Calculate the value function $V(s)$ for the current policy; once the delta is below the required threshold, the loop can stop (implementation is done using a `while` loop). [cite: 34]
2.  [cite_start]**Policy Improvement** – For each state, choose an action that leads to the maximum future reward, based on the previously calculated value function. [cite: 35]

[cite_start]If at least one change occurs in the policy in the same iteration, the algorithm will continue to examine improvements until it receives 2 identical policies consecutively. [cite: 36]

### [cite_start]State Space: [cite: 37]

[cite_start]The implementation of the states is as follows: [cite: 38]
[cite_start]`state:(row, col, has_bag, has_rope)` [cite: 39]
[cite_start]This implementation effectively creates a policy with four different layers: [cite: 39]

* [cite_start]Action when the agent has no items. [cite: 40]
* [cite_start]Action when the agent only has the bag. [cite: 41]
* [cite_start]Action when the agent only has the rope. [cite: 42]
* [cite_start]Action when the agent has both. [cite: 43]

**Understanding the Order:** The agent was not programmed to collect the bag first. [cite_start]It learned this on its own. [cite: 44] [cite_start]The reward system is structured such that if it collects the rope before the bag, or reaches the end point without both items, it will be penalized. [cite: 45] [cite_start]The system gave it a reward for collecting the rope only if it already had the bag. [cite: 45] [cite_start]Therefore, in its calculations, it discovered that the path where it collects the bag, then the rope, and only then reaches the endpoint, yields a significantly higher final score than any other path. [cite: 46]

### [cite_start]Rewards: [cite: 47]

* [cite_start]For collecting the bag, the agent receives 20 points. [cite: 48]
* [cite_start]For collecting the rope without the bag, the agent receives a penalty of -10 points, but if it already holds the bag, it receives a reward of 30 points. [cite: 49]
* [cite_start]For reaching the exit without both items, the agent is penalized -20 points, and if it reaches the end with both, it receives a reward of 100 points. [cite: 50]

### [cite_start]Controllable Parameters: [cite: 51]

* [cite_start]**Discount Factor (Gamma)** – Importance of future rewards (default 0.9). [cite: 52]
* [cite_start]**Theta** – Convergence threshold for the policy evaluation step (the threshold below which delta must fall) (default 0.000001). [cite: 53]
* [cite_start]**Starting Items** – Can be set whether the agent starts with the bag, rope, or both. [cite: 54]
* [cite_start]**Number of walls or slippery tiles** [cite: 55]

## [cite_start]Room 2: SARSA [cite: 56]

[cite_start]An implementation of the SARSA algorithm, where the agent is aware that it will not always act optimally and is exploratory by nature, so its next step might be random. [cite: 57] [cite_start]Such an implementation creates a situation where the agent is more "aware" of its limitations and that it might "slip," and therefore it learns safer and more conservative approaches, especially near environmental hazards. [cite: 58]

### [cite_start]The Challenge: [cite: 59]

[cite_start]In this room, the grid is divided into 3 separate areas. [cite: 60] [cite_start]In the first area, an enemy patrols a fixed rectangular path around the first area. [cite: 60] [cite_start]The agent needs to learn to collect the key located in this area, which opens a random wall separating the first and second spaces. [cite: 61] [cite_start]It needs to learn to move to the second space and enter there through a tunnel that will lead it to the third and final space, where the exit point is located. [cite: 62]

[cite_start]In this room, walls and slippery tiles are randomly distributed (in this room, the chance of slipping in any legal direction is equal for all directions, and so is in the next room), but the general structure of the three spaces is fixed. [cite: 63] [cite_start]The path by which the enemy patrols is fixed and will always include only regular tiles (and the starting square). [cite: 64]

[cite_start]It's possible that an unsolvable map will be generated for the agent, for example, if the passage between the first and second spaces is blocked by a random wall, or if the way to the tunnel is blocked. [cite: 65] [cite_start]The same applies to the third space; it's possible that the path from the tunnel exit to the end square is also blocked. [cite: 66]

### [cite_start]Learning Method: [cite: 67]

[cite_start]The SARSA algorithm is Model-Free and On-Policy: [cite: 68]

* [cite_start]**Model-Free** – Meaning it learns through actual experience, without needing to know transition probabilities. [cite: 69]
* [cite_start]**On-Policy** – Meaning it learns the values of the current policy, including random actions. [cite: 70] [cite_start]The algorithm updates the Q-value based on $(s, a, r, s', a')$ where $a'$ is the action actually chosen in the next state. [cite: 70]

### [cite_start]State Space: [cite: 71]

[cite_start]The implementation of the states is as follows: [cite: 72]
[cite_start]`state:(row, col, has_key)` [cite: 73]
[cite_start]This implementation effectively creates a policy with two different layers: [cite: 73]

* [cite_start]Action when the agent does not have the key. [cite: 74]
* [cite_start]Action when the agent has collected the key. [cite: 75]

[cite_start]Learning is performed in such a way that the agent starts each episode from the same starting point, understanding that if it starts randomly in the second or third space, it is in an illogical state, as it starts without a key, creating a situation where it might start in one of these spaces in an $(r,c,0)$ state, which is illogical and its learning is redundant. [cite: 76]

### [cite_start]Rewards: [cite: 77]

* [cite_start]For collecting the key, the agent receives a reward of 50 points. [cite: 78]
* [cite_start]For using the tunnel, it receives a reward of 5 points. [cite: 79]
* [cite_start]For reaching the exit, it receives a reward of 100 points. [cite: 80]
* [cite_start]For colliding with a wall or attempting to "exit" the grid, the agent is penalized -5 points. [cite: 81]
* [cite_start]If the agent is caught by the enemy, it receives a penalty of -100 points, and the episode ends. [cite: 82]

### [cite_start]Controllable Parameters: [cite: 83]

* [cite_start]**Alpha** – Learning rate (default 0.1). [cite: 84]
* [cite_start]**Epsilon** – Exploration rate (default 1). [cite: 85]
* [cite_start]**Number of Episodes** (default 5000). [cite: 86]
* [cite_start]**Maximum steps per episode** (default 200). [cite: 87]
* [cite_start]**Minimum Epsilon** (0.01). [cite: 88]
* [cite_start]**Decay Factor** (0.9995). [cite: 89]
* [cite_start]**Number of walls and slippery tiles** [cite: 90]

## [cite_start]Room 3: Q-Learning [cite: 91]

[cite_start]The third agent is the "greedy explorer." [cite: 92] [cite_start]Like the agent in Room 2, it learns through trial and error, but it is always optimistic and assumes its next step will be optimal. [cite: 92] [cite_start]This means it does not assume it might "slip" and therefore always calculates the reward based on the optimal action. [cite: 93]

### [cite_start]The Challenge: Solving a Complex Physics Puzzle and Dealing with "Fear" [cite: 94]

[cite_start]In this room, there are two "islands" in the center of the grid. [cite: 95] [cite_start]An island is a square surrounded by pits, and falling into a pit ends the episode and incurs a penalty for the agent. [cite: 95] [cite_start]In the center of the island there is a legal square, with a key on it. [cite: 96] [cite_start]The agent must collect 2 keys (order doesn't matter) to open the door blocking the exit. [cite: 96]

[cite_start]There are 2 planks scattered across the grid. [cite: 97] [cite_start]The agent can pull or push the planks, and must understand that it can push the planks into a pit, and with the help of the rope (not really, just for the game's story) and the plank, create a bridge that will allow it to collect the key. [cite: 97] [cite_start]This is a significant challenge because the reward for building the bridge only comes after a long series of actions, each of which has no immediate reward. [cite: 98] [cite_start]Therefore, the default map layout is without any additional obstacles. [cite: 99]

[cite_start]The map can be edited to make the agent's challenge harder by adding walls. [cite: 100] [cite_start]This can lead to the generation of maps that are unsolvable or very difficult to solve, and therefore in such a situation, it is recommended to increase the number of episodes significantly (around 40,000, or even more). [cite: 100]

[cite_start]In this room, there were many challenges regarding model training, the main one being the local optimum problem. [cite: 101] [cite_start]This problem arose due to the agent's "fear." [cite: 101] [cite_start]It learned that pushing the plank gives a small and safe reward (+5), while approaching a pit is very dangerous (-100). [cite: 102] [cite_start]Therefore, it preferred to get stuck in the "OK" and safe solution, rather than risk finding the perfect solution. [cite: 103] [cite_start]And so it began pushing the plank in loops until the steps ran out. [cite: 104] [cite_start]The solution was to remove the reward for pushing the plank and add a small additional penalty for each action. [cite: 105]

### [cite_start]Learning Method: Q-Learning [cite: 106]

[cite_start]The Q-Learning algorithm is used here – it is also model-free, but Off-Policy. [cite: 107]

* [cite_start]**Off-Policy** – The algorithm learns the optimal policy regardless of actual actions. [cite: 108] [cite_start]It uses the maximum Q-value of the next state, assuming the agent will choose the best possible action. [cite: 108]

### [cite_start]State Space: [cite: 109]

[cite_start]The implementation of the states is as follows: [cite: 110]
[cite_start]`state:(row, col, row_plank1, col_plank1, row_plank2, col_plank2, has_silver_key, has_gold_key)` [cite: 111]

[cite_start]This means there is a rather large state space, so that the agent "understands" that a bridge has indeed been created and the plank can no longer be pushed. [cite: 112] [cite_start]When a plank is pushed into a pit, its row value becomes -1, and the agent receives a reward for this (positive or negative depending on the bridge's location). [cite: 112] [cite_start]When the value becomes -1, the agent learns that the square into which it pushed the plank has become legal and it can now move on it to collect the key. [cite: 113]

### [cite_start]Rewards: [cite: 114]

* [cite_start]For collecting a key, the agent receives a reward of 50 points. [cite: 115]
* [cite_start]For building a bridge in a location that allows key collection, it receives a reward of 30 points. [cite: 116]
* [cite_start]For building a bridge in a location that does not allow key collection, it receives a penalty of -20 points. [cite: 117]
* [cite_start]For building a bridge to an island that already has a bridge, it receives a penalty of -50 points. [cite: 118]
* [cite_start]For reaching the exit, it receives a reward of 100 points. [cite: 119]
* [cite_start]For colliding with a wall or attempting to "exit" the grid, the agent is penalized -5 points. [cite: 120]
* [cite_start]If the agent falls into a pit, it receives a penalty of -100 points, and the episode ends. [cite: 121]
* [cite_start]The agent is also penalized -0.1 points for each step it takes. [cite: 122] [cite_start]The reason this addition is crucial for implementation is because the agent might find a "safe path" but one without rewards and prefer this path because in this room's structure, it is mainly penalized at the beginning, as reaching an initial reward is a significant challenge. [cite: 122] [cite_start]Therefore, a path where it performs the same action in a loop to avoid being penalized during exploration is prevented. [cite: 123]

### [cite_start]Controllable Parameters: [cite: 124]

* [cite_start]**Alpha** – Learning rate (default 0.1). [cite: 125]
* [cite_start]**Epsilon** – Exploration rate (default 1). [cite: 126]
* [cite_start]**Number of Episodes** (default 5000). [cite: 127]
* [cite_start]**Maximum steps per episode** (default 200). [cite: 128]
* [cite_start]**Minimum Epsilon** (0.01). [cite: 129]
* [cite_start]**Decay Factor** (0.9995). [cite: 130]
* [cite_start]**Number of walls** [cite: 131]
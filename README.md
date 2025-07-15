# Project Documentation

## Important: Required Libraries for Running: pygame, numpy, pandas, matplotlib

[In addition, recordings of rooms 1-3.](https://drive.google.com/file/d/1y35IOVfNC3u4EAr5YaBeZice1di9FXD1/view?usp=drive_link)

## Installation and Running the Game:

1.  **Extract the project folder to your computer.**
2.  **Navigate to the folder via PowerShell.**
3.  **Open a virtual environment to contain the exact environment.**
    ```bash
    py -3.11 -m venv venv
    ```
4.  **Run the command that activates the virtual environment.**
    ```bash
    ./venv/Scripts/Activate.ps1
    ```
5.  **You might need to run the following command beforehand.**
    ```bash
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    ```
6.  **Install the required libraries.**
    ```bash
    pip install -r requirements.txt
    ```
7.  **Run the main project file.**
    ```bash
    python main.py
    ```
    This command will activate the rooms.

## Usage Instructions and Screens:

* When the game loads, the grid appears with the first room, and next to the grid, there's a console with messages about the actions taken in the game.
* In each room, it's possible to refresh the room and get a new layout that is randomly initialized (unsolvable room iterations may occur).
* Each room has the option to edit the settings of that space and the agent's learning method.
* Each agent has the option to train "slowly" or "quickly." In slow mode, iterations can be seen in Room 1, and in Rooms 2 and 3, episodes can be observed occurring.
* In Rooms 2-3, if slow training is chosen, there is also an option to skip to any episode within the training range (e.g., if training 5000 episodes, you can skip to episodes 2-4999).
* The same episode can be run to observe the agent's learning state at that point.
* Also, in all rooms, the final training can be run by clicking the `Run` button.
* Animation can be paused with the `Pause` button, and the run can be reset with the `Reset Run` button.
* In Rooms 2,3, you can click the `Show Q-Vals` button; this button is a toggle, and it will display the trained Q-table for that state on the grid.
* In Rooms 2,3, you can click the `Plot Data` button, which will open a window (it's recommended to maximize it to full screen for comfortable viewing of the buttons) where you can see 3 different graphs:
    * The first relates to rewards during training.
    * The second relates to the number of steps during training.
    * The third shows the grid and how many steps the agent took in each direction for each tile.

## Room 1: Dynamic Programming

The first agent uses Dynamic Programming, possessing a model of the environment and its rules. It is not required to explore the environment but rather to calculate the ideal policy using the given model to meet the task according to the rules. It starts with some random policy and then begins calculations to perform Policy Iteration.

### The Challenge:

The agent must navigate the grid to first collect the bag, then the rope, and finally reach the exit. The main challenge is the presence of wall tiles and slippery tiles. On slippery tiles, the action chosen by the agent is not significant, and the outcome is determined by a fixed probability (randomly rolled with each new map generation, where the minimum threshold for a legal direction, i.e., not towards a wall or outside the grid, is 20%, to prevent very low chances in certain directions). The agent is forced to develop a policy that accounts for this uncertainty.

Slippery tiles posed a problem in implementation in terms of how the direction appeared when the agent calculated the policy, because its decision has no meaning, and therefore there is no real meaning to the direction it "would most want" to move in. Therefore, I added a mechanism for these tiles, where even though the choice is not important, the displayed policy will be such that when the policy converges, the arrows appearing on the tiles (i.e., where the agent should want to go) will point to the closest legal tile that has the highest expected reward value.

Additionally, I limited the random placement of items relative to the center of the map.

### Learning Method: Policy Iteration

The algorithm repeats two steps until the policy stabilizes:

1.  **Policy Evaluation** – Calculate the value function $V(s)$ for the current policy; once the delta is below the required threshold, the loop can stop (implementation is done using a `while` loop).
2.  **Policy Improvement** – For each state, choose an action that leads to the maximum future reward, based on the previously calculated value function.

If at least one change occurs in the policy in the same iteration, the algorithm will continue to examine improvements, until it receives 2 identical policies consecutively.

### State Space:

The implementation of the states is as follows:
`state:(row, col, has_bag, has_rope)`
This implementation effectively creates a policy with four different layers:

* Action when the agent has no items.
* Action when the agent only has the bag.
* Action when the agent only has the rope.
* Action when the agent has both.

**Understanding the Order:** The agent was not programmed to collect the bag first. It learned this on its own. The reward system is structured such that if it collects the rope before the bag, or reaches the end point without both items, it will be penalized. The system gave it a reward for collecting the rope only if it already had the bag. Therefore, in its calculations, it discovered that the path where it collects the bag, then the rope, and only then reaches the endpoint, yields a significantly higher final score than any other path.

### Rewards:

* For collecting the bag, the agent receives 20 points.
* For collecting the rope without the bag, the agent receives a penalty of -10 points, but if it already holds the bag, it receives a reward of 30 points.
* For reaching the exit without both items, the agent is penalized -20 points, and if it reaches the end with both, it receives a reward of 100 points.

### Controllable Parameters:

* **Discount Factor (Gamma)** – Importance of future rewards (default 0.9).
* **Theta** – Convergence threshold for the policy evaluation step (the threshold below which delta must fall) (default 0.000001).
* **Starting Items** – Can be set whether the agent starts with the bag, rope, or both.
* **Number of walls or slippery tiles**.

## Room 2: SARSA

An implementation of the SARSA algorithm, where the agent is aware that it will not always act optimally and is exploratory by nature, so its next step might be random. Such an implementation creates a situation where the agent is more "aware" of its limitations and that it might "slip," and therefore it learns safer and more conservative approaches, especially near environmental hazards.

### The Challenge:

In this room, the grid is divided into 3 separate areas. In the first area, an enemy patrols a fixed rectangular path around the first area. The agent needs to learn to collect the key located in this area, which opens a random wall separating the first and second spaces. It needs to learn to move to the second space and enter there through a tunnel that will lead it to the third and final space, where the exit point is located.

In this room, walls and slippery tiles are randomly distributed (in this room, the chance of slipping in any legal direction is equal for all directions, and so is in the next room), but the general structure of the three spaces is fixed. The path by which the enemy patrols is fixed and will always include only regular tiles (and the starting square).

It's possible that an unsolvable map will be generated for the agent, for example, if the passage between the first and second spaces is blocked by a random wall, or if the way to the tunnel is blocked. The same applies to the third space; it's possible that the path from the tunnel exit to the end square is also blocked.

### Learning Method:

The SARSA algorithm is Model-Free and On-Policy:

* **Model-Free** – Meaning it learns through actual experience, without needing to know transition probabilities.
* **On-Policy** – Meaning it learns the values of the current policy, including random actions. The algorithm updates the Q-value based on $(s, a, r, s', a')$ where $a'$ is the action actually chosen in the next state.

### State Space:

The implementation of the states is as follows:
`state:(row, col, has_key)`
This implementation effectively creates a policy with two different layers:

* Action when the agent does not have the key.
* Action when the agent has collected the key.

Learning is performed in such a way that the agent starts each episode from the same starting point, understanding that if it starts randomly in the second or third space, it is in an illogical state, as it starts without a key, creating a situation where it might start in one of these spaces in an $(r,c,0)$ state, which is illogical and its learning is redundant.

### Rewards:

* For collecting the key, the agent receives a reward of 50 points.
* For using the tunnel, it receives a reward of 5 points.
* For reaching the exit, it receives a reward of 100 points.
* For colliding with a wall or attempting to "exit" the grid, the agent is penalized -5 points.
* If the agent is caught by the enemy, it receives a penalty of -100 points, and the episode ends.

### Controllable Parameters:

* **Alpha** – Learning rate (default 0.1).
* **Epsilon** – Exploration rate (default 1).
* **Number of Episodes** (default 5000).
* **Maximum steps per episode** (default 200).
* **Minimum Epsilon** (0.01).
* **Decay Factor** (0.9995).
* **Number of walls and slippery tiles**.

## Room 3: Q-Learning

The third agent is the "greedy explorer". Like the agent in Room 2, it learns through trial and error, but it is always optimistic and assumes its next step will be optimal. This means it does not assume it might "slip" and therefore always calculates the reward based on the optimal action.

### The Challenge: Solving a Complex Physics Puzzle and Dealing with "Fear"

In this room, there are two "islands" in the center of the grid. An island is a square surrounded by pits, and falling into a pit ends the episode and incurs a penalty for the agent. In the center of the island there is a legal square, with a key on it. The agent must collect 2 keys (order doesn't matter) to open the door blocking the exit.

There are 2 planks scattered across the grid. The agent can pull or push the planks, and must understand that it can push the planks into a pit, and with the help of the rope (not really, just for the game's story) and the plank, create a bridge that will allow it to collect the key. This is a significant challenge because the reward for building the bridge only comes after a long series of actions, each of which has no immediate reward. Therefore, the default map layout is without any additional obstacles.

The map can be edited to make the agent's challenge harder by adding walls. This can lead to the generation of maps that are unsolvable or very difficult to solve, and therefore in such a situation, it is recommended to increase the number of episodes significantly (around 40,000, or even more).

In this room, there were many challenges regarding model training, the main one being the local optimum problem, this problem arose due to the agent's "fear". It learned that pushing the plank gives a small and safe reward (+5), while approaching a pit is very dangerous (-100). Therefore, it preferred to get stuck in the "OK" and safe solution, rather than risk finding the perfect solution. And so it began pushing the plank in loops until the steps ran out. The solution was to remove the reward for pushing the plank, and add a small additional penalty for each action.

### Learning Method: Q-Learning

The Q-Learning algorithm is used here – it is also model-free, but Off-Policy.

* **Off-Policy** – The algorithm learns the optimal policy regardless of actual actions. It uses the maximum Q-value of the next state, assuming the agent will choose the best possible action.

### State Space:

The implementation of the states is as follows:
`state:(row, col, row_plank1, col_plank1, row_plank2, col_plank2, has_silver_key, has_gold_key)`

This means there is a rather large state space, so that the agent "understands" that a bridge has indeed been created and the plank can no longer be pushed. When a plank is pushed into a pit, its row value becomes -1, and the agent receives a reward for this (positive or negative depending on the bridge's location). When the value becomes -1, the agent learns that the square into which it pushed the plank has become legal and it can now move on it to collect the key.

### Rewards:

* For collecting a key, the agent receives a reward of 50 points.
* For building a bridge in a location that allows key collection, it receives a reward of 30 points.
* For building a bridge in a location that does not allow key collection, it receives a penalty of -20 points.
* For building a bridge to an island that already has a bridge, it receives a penalty of -50 points.
* For reaching the exit, it receives a reward of 100 points.
* For colliding with a wall or attempting to "exit" the grid, the agent is penalized -5 points.
* If the agent falls into a pit, it receives a penalty of -100 points, and the episode ends.
* The agent is also penalized -0.1 points for each step it takes. The reason this addition is crucial for implementation is because the agent might find a "safe path" but one without rewards and prefer this path because in this room's structure, it is mainly penalized at the beginning, as reaching an initial reward is a significant challenge. Therefore, a path where it performs the same action in a loop to avoid being penalized during exploration is prevented.

### Controllable Parameters:

* **Alpha** – Learning rate (default 0.1).
* **Epsilon** – Exploration rate (default 1).
* **Number of Episodes** (default 5000).
* **Maximum steps per episode** (default 200).
* **Minimum Epsilon** (0.01).
* **Decay Factor** (0.9995).
* **Number of walls**.

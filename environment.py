import numpy as np
import random

class GridWorld:
    def __init__(self, size=20, start_pos=(0,0), goal_pos=(19,19), num_obstacles=20):
        self.size = size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.num_obstacles = num_obstacles

        self.agent_pos = None
        self.obstacles = []
        self._generate_obstacles()

        self.action_space = [0, 1, 2, 3] # 0:up, 1:down, 2:left, 3:right
        self.action_space_size = len(self.action_space)
        self.state_size = 2 # (row, col)

        self.reset()

    def _generate_obstacles(self):
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            obstacle_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if obstacle_pos != self.start_pos and obstacle_pos != self.goal_pos and obstacle_pos not in self.obstacles:
                self.obstacles.append(obstacle_pos)

    def reset(self):
        self.agent_pos = self.start_pos
        return self.get_state()

    def get_state(self):
        return self.agent_pos # Represented as (row, col)

    def get_action_space_size(self):
        return self.action_space_size

    def get_state_size(self): #  This might be more about the shape for the NN
        return self.state_size 

    def step(self, action):
        prev_pos = self.agent_pos
        row, col = self.agent_pos

        if action == 0: # Up
            row -= 1
        elif action == 1: # Down
            row += 1
        elif action == 2: # Left
            col -= 1
        elif action == 3: # Right
            col += 1
        
        next_pos = (row, col)
        reward = -0.1 # Cost of moving
        done = False

        # Check boundaries
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            reward = -50 # Penalty for hitting a wall
            done = True
            next_pos = prev_pos # Agent bounces back
        elif next_pos == self.goal_pos:
            reward = 50 # Reward for reaching the goal
            done = True
            self.agent_pos = next_pos
        elif next_pos in self.obstacles:
            reward = -50 # Penalty for hitting an obstacle
            done = True
            next_pos = prev_pos # Agent bounces back (or stays, depending on desired behavior)
        else:
            self.agent_pos = next_pos

        return self.get_state(), reward, done

    def render(self, clear_screen=False):
        if clear_screen:
            # Basic clear screen for command line, might need adjustment for different OS
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # Mark obstacles
        for obs_r, obs_c in self.obstacles:
            grid[obs_r][obs_c] = 'X'
        
        # Mark goal
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        
        # Mark agent
        if self.agent_pos:
            grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
            
        print(f"Agent @ {self.agent_pos} | Goal @ {self.goal_pos}")
        for r in range(self.size):
            print(' '.join(grid[r]))
        print("-" * (self.size * 2))

if __name__ == '__main__':
    # Example Usage
    env = GridWorld(size=10, num_obstacles=5, goal_pos=(9,9))
    env.render()
    
    state = env.reset()
    print(f"Initial State: {state}")
    
    # Take a few random steps
    for i in range(5):
        action = random.choice(env.action_space)
        print(f"Taking action: {['Up', 'Down', 'Left', 'Right'][action]}")
        next_state, reward, done = env.step(action)
        env.render()
        print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            print("Episode finished.")
            break
        state = next_state

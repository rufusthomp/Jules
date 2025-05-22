import numpy as np
from environment import GridWorld
from agent import DQNAgent
import time # To control rendering speed

def state_tuple_to_numpy(state_tuple):
    """Converts (row, col) tuple to a numpy array [row, col]."""
    return np.array(state_tuple)

if __name__ == "__main__":
    # Environment parameters
    env_size = 20
    start_pos = (0,0)
    goal_pos = (env_size-1, env_size-1)
    num_obstacles = 20 

    # Agent parameters
    state_size = 2 # row, col
    action_size = 4 # up, down, left, right

    # Training parameters
    EPISODES = 1000 
    BATCH_SIZE = 64 
    MAX_STEPS_PER_EPISODE = 200 

    # Initialize environment and agent
    env = GridWorld(size=env_size, start_pos=start_pos, goal_pos=goal_pos, num_obstacles=num_obstacles)
    agent = DQNAgent(state_size=state_size, action_size=action_size,
                     learning_rate=0.001, discount_factor=0.99,
                     exploration_rate=1.0, exploration_decay=0.999, 
                     min_exploration_rate=0.01, replay_buffer_size=20000)

    print(f"Starting training for {EPISODES} episodes...")
    print(f"Environment: {env_size}x{env_size} grid, Goal: {goal_pos}, Obstacles: {num_obstacles}")
    print(f"Agent: Epsilon start={agent.epsilon:.2f}, decay={agent.epsilon_decay:.3f}, min={agent.epsilon_min}")
    print(f"Replay Batch Size: {BATCH_SIZE}, Max Steps/Episode: {MAX_STEPS_PER_EPISODE}")

    episode_rewards_history = []

    for e in range(EPISODES):
        current_state_tuple = env.reset()
        current_state_np = state_tuple_to_numpy(current_state_tuple)
        
        total_reward_episode = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(current_state_np) # Pass 1D numpy array
            
            next_state_tuple, reward, done = env.step(action)
            next_state_np = state_tuple_to_numpy(next_state_tuple)
            
            agent.remember(current_state_np, action, reward, next_state_np, done) # Store 1D numpy arrays
            
            current_state_tuple = next_state_tuple
            current_state_np = next_state_np
            total_reward_episode += reward
            
            # Render the environment (less frequently for larger grids/longer training)
            if e % 50 == 0 or EPISODES - e <= 10 : 
                if step % 10 == 0 or done : 
                    env.render(clear_screen=True)
                    print(f"Episode: {e+1}/{EPISODES}, Step: {step+1}/{MAX_STEPS_PER_EPISODE}, Action: {['Up','Down','Left','Right'][action]}")
                    print(f"State: {current_state_tuple}, Reward: {reward:.1f}, Total Reward: {total_reward_episode:.2f}, Done: {done}")
                    print(f"Epsilon: {agent.epsilon:.4f}")
                    time.sleep(0.05) 

            if done:
                break 

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
        
        episode_rewards_history.append(total_reward_episode)

        print(f"Episode: {e+1}/{EPISODES} | Total Reward: {total_reward_episode:.2f} | Steps: {step+1} | Epsilon: {agent.epsilon:.4f}")

        moving_avg_window = 20
        current_episode_count = len(episode_rewards_history)
        actual_window_size = min(current_episode_count, moving_avg_window)
        
        if actual_window_size > 0:
            moving_avg_reward = np.mean(episode_rewards_history[-actual_window_size:])
            print(f"Moving Avg Reward (last {actual_window_size} episodes): {moving_avg_reward:.2f}")

        if e % 100 == 0 and e > 0: # This condition won't be met with EPISODES = 5
            try:
                agent.save(f"gridworld-dqn-episode-{e}.weights.h5")
                print(f"Saved model weights at episode {e}")
            except Exception as ex:
                print(f"Error saving model at episode {e}: {ex}")


    print("Training finished.")
    try:
        agent.save("gridworld-dqn-final.weights.h5")
        print("Saved final model weights.")
    except Exception as ex:
        print(f"Error saving final model: {ex}")

    print("\nRunning a few episodes in evaluation mode (epsilon=0)...")
    agent.epsilon = 0 # Turn off exploration
    agent.epsilon_min = 0 # Ensure no exploration even if min_epsilon was >0 during training
    
    for eval_episode in range(5): # Evaluation episodes
        current_state_tuple = env.reset()
        print(f"\nEvaluation Episode: {eval_episode+1} - Starting at: {current_state_tuple}")
        current_state_np = state_tuple_to_numpy(current_state_tuple)
        total_reward_eval = 0
        
        for step in range(MAX_STEPS_PER_EPISODE): # Use same max steps for safety
            env.render(clear_screen=True)
            action = agent.act(current_state_np) 
            
            next_state_tuple, reward, done = env.step(action)
            
            print(f"Step: {step+1}, State: {current_state_tuple}, Action: {['Up','Down','Left','Right'][action]}, Reward: {reward:.1f}, Next: {next_state_tuple}, Done: {done}")
            time.sleep(0.2) # Slower for observation

            current_state_tuple = next_state_tuple
            current_state_np = state_tuple_to_numpy(next_state_tuple)
            total_reward_eval += reward
            if done:
                env.render(clear_screen=False) # Show final state
                print(f"Goal reached or episode ended at {current_state_tuple}!")
                break
        print(f"Evaluation Episode: {eval_episode+1}, Total Reward: {total_reward_eval:.2f}, Steps: {step+1}")
    
    print("\nEvaluation complete.")

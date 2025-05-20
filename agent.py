import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, 
                 learning_rate=0.001, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.999, 
                 min_exploration_rate=0.01, replay_buffer_size=20000):
        self.state_size = state_size  # e.g., 2 for (row, col)
        self.action_size = action_size # e.g., 4 for up, down, left, right
        
        self.memory = deque(maxlen=replay_buffer_size)
        
        self.learning_rate = learning_rate
        self.gamma = discount_factor # discount_factor
        
        self.epsilon = exploration_rate # exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep Q-learning Model
        model = Sequential()
        # Input layer: expects flattened state if state is multi-dimensional, 
        # or just state_size if it's already 1D.
        # Our state is (row, col), which is simple enough.
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) # Output Q-values for each action
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore: random action
        
        # Exploit: predict optimal action based on current Q-values
        # State needs to be in the correct shape for the model, e.g., (1, state_size)
        state_reshaped = np.reshape(state, [1, self.state_size])
        # The following line was corrected to handle TensorFlow 2.x behavior with predict
        act_values = self.model.predict(state_reshaped, verbose=0)
        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return # Not enough memories to replay

        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Predict Q-values for current states and next states
        # For current states, these are the Q-values we will update
        # The following lines were corrected to handle TensorFlow 2.x behavior with predict
        current_q_values_batch = self.model.predict(states, verbose=0)
        # For next states, these are used to calculate the target Q-value
        next_q_values_batch = self.model.predict(next_states, verbose=0)

        # Prepare the target Q-values for training
        # This will be the same shape as current_q_values_batch
        target_q_values_batch = np.copy(current_q_values_batch)

        for i in range(batch_size):
            # state = states[i] # Not directly used in this loop structure for Bellman
            action = actions[i]
            reward = rewards[i]
            # next_state = next_states[i] # Not directly used in this loop structure for Bellman
            done = dones[i]
            
            current_q_values_for_state = current_q_values_batch[i] # Q-values for the current state in the batch
            
            if done:
                target_q_for_action = reward
            else:
                # Bellman equation: Q(s,a) = r + gamma * max_a'(Q(s',a'))
                target_q_for_action = reward + self.gamma * np.amax(next_q_values_batch[i])
            
            # Update only the Q-value for the action taken
            current_q_values_for_state[action] = target_q_for_action
            target_q_values_batch[i] = current_q_values_for_state


        # Train the model: states are inputs, target_q_values_batch are the desired outputs
        self.model.fit(states, target_q_values_batch, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == '__main__':
    # Example Usage (requires a mock environment or integration with GridWorld)
    mock_state_size = 2
    mock_action_size = 4
    agent = DQNAgent(mock_state_size, mock_action_size)
    print(f"Agent initialized with model: ")
    agent.model.summary()

    # Mock a state
    mock_state = (0, 0) # Example state (row, col)
    
    # Test act method
    # Ensure mock_state is a NumPy array for the agent's act method
    action = agent.act(np.array(mock_state))
    print(f"Agent chose action: {action} from state: {mock_state}")

    # Test remember and replay (simplified)
    # Ensure states are NumPy arrays for remember method
    mock_next_state = (0, 1)
    mock_reward = 1
    mock_done = False
    agent.remember(np.array(mock_state), action, mock_reward, np.array(mock_next_state), mock_done)
    # Add more diverse experiences for better replay testing
    agent.remember(np.array((0,1)), 2, -1, np.array((1,1)), False)
    agent.remember(np.array((1,1)), 3, 10, np.array((1,2)), True)
    
    print(f"Memory size: {len(agent.memory)}")

    if len(agent.memory) >= 2: # Ensure enough samples for batch_size=2 for demo
        agent.replay(2) 
        print("Agent replayed experiences (batch_size=2).")
    
    print(f"Agent epsilon after potential replay: {agent.epsilon}")
    
    # Test save/load
    try:
        agent.save("mock_agent_weights.h5")
        print("Mock agent weights saved.")
        agent.load("mock_agent_weights.h5")
        print("Mock agent weights loaded.")
    except Exception as e:
        print(f"Error during save/load: {e}. This might happen if h5py is not available or due to TF version issues in some environments.")

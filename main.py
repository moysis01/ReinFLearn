import gym
from env.warehouse_env import WarehouseEnv  # Custom environment
from models.dqn_agent import DQNAgent       # DQN agent
from utils.helpers import plot_rewards      # Plotting helper

def main():
    # Create the environment
    env = WarehouseEnv(max_inventory=100, max_orders=100, restock_amount=10)
    
    # Create the DQN agent
    agent = DQNAgent(env)
    
    # Training variables
    episodes = 500
    rewards = []
    
    # Train the agent
    for episode in range(episodes):
        state = env.reset()
        state = state.reshape(1, -1)  # Reshape state for network input
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, -1)  # Reshape next state for network input
            agent.remember(state, action, reward, next_state, done)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}")
    
    # Plot the rewards
    plot_rewards(rewards)

if __name__ == "__main__":
    main()

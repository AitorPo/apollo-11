from agent import Agent
from constants import state_size, action_size
from train import train, deserialize_model_params, create_video_of_agent


if __name__ == '__main__':
    agent = Agent(state_size, action_size)
    train(agent)
    deserialize_model_params()
    create_video_of_agent(agent, 'LunarLander-v2')

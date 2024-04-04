import torch
import numpy as np

from constants import scores, number_episodes, max_steps_per_episode, env, epsilon_end_value, epsilon_decay_value
import gymnasium as gym
import imageio


def perform_save_params_and_print_results(episode_number, score, model_params, epsilon_):
    print(f"\rEpisode nº {episode_number}\tAverage Score: {score:.4f}", end="")
    if episode_number % 100 == 0:
        print(f"\rEpisode {episode_number}\tAverage Score: {score:.4f}")
    if np.mean(scores) >= 200.0:
        print(f"\rGame completed in {episode_number:d} episodes\tAverage Score: {score:.4f}")
        print("Saving model parameters...")
        torch.save(model_params, 'checkpoint.pt')
        return True


def deserialize_model_params():
    return print(torch.load('checkpoint.pt'))


def create_video_of_agent(agent, env_name):
    agent_env = gym.make(env_name, render_mode='rgb_array')
    state, _ = agent_env.reset()
    done = False
    frames = []
    while not done:
        frame = agent_env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = agent_env.step(action.item())
    agent_env.close()
    imageio.mimsave('apollo-11.mp4', frames, fps=30)


def train(agent):
    epsilon = 1.0
    for episode in range(1, number_episodes + 1):
        # We want to reset the environment at the start of each episode
        state, _ = env.reset()
        score = 0
        for step in range(max_steps_per_episode):
            # Playing the action
            selected_action = agent.act(state, epsilon)
            # Each time the agent plays an action it receives a reward
            next_state, reward, done, _, _ = env.step(selected_action)
            agent.step(state, selected_action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        epsilon = max(epsilon_end_value, epsilon_decay_value * epsilon)
        is_finished = perform_save_params_and_print_results(episode, np.mean(scores),
                                                            agent.local_q_nnetwork.state_dict(), epsilon)
        if is_finished:
            break

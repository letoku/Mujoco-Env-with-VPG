from structures.training_manager import TrainingManager
from structures.agent import Agent
from car_wall_env import CarWallEnv
import torch


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    env = CarWallEnv(render=False)
    # vis_env = CarWallEnv(render=True)
    train_manager = TrainingManager()
    sample_size = 20
    agent = Agent(env, states_dim=2, actions_dim=2, sample_size=sample_size, vis_env=None)
    iteration = 0
    max_iterations = 500

    used_trained = False
    if used_trained:
        import pickle
        with open('wall_env_braking_agent_policy.pkl', 'rb') as handle:
            trained_policy = pickle.load(handle)
            agent.policy = trained_policy

    agent.plot_policy_map()
    if not used_trained:
        while not train_manager.converged():
            agent.clear_memory()
            for i in range(sample_size):
                agent.collect_trajectory()
            avg_score = agent.update()
            train_manager.update_after_iteration(avg_score)

            print(f'Avg score on {iteration} iteration: {avg_score}')
            iteration += 1
            if iteration > max_iterations:
                break

        train_manager.plot_scores()

    """see in viewer"""
    visualization_env = CarWallEnv(render=True)
    agent.env = visualization_env

    for i in range(5):
        agent.collect_trajectory(greedy=True)

    """record videos"""
    visualization_env = CarWallEnv(render=True, off_screen=True)
    agent.env = visualization_env

    for i in range(5):
        agent.collect_trajectory(greedy=True, record_video=True)

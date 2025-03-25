import numpy as np
import torch
from structures.network import Policy
from datetime import datetime


class Agent:
    def __init__(self, env, states_dim, actions_dim, sample_size: int, learning_rate: float = 5e-3, vis_env=None):
        self.env = env
        self.vis_env = vis_env
        self.policy = Policy(input_dim=states_dim, output_dim=actions_dim)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.trajectories = []

        self.batch_states = []
        self.batch_actions = []
        self.batch_weights = []
        self.sample_size = sample_size

    def clear_memory(self):
        self.trajectories = []
        self.batch_states = []
        self.batch_actions = []
        self.batch_weights = []

    def get_greedy_action(self, state):
        return torch.argmax(self.policy(torch.tensor(state)).logits).item()

    def get_action(self, state):
        return self.policy(torch.tensor(state)).sample(sample_shape=[1]).detach().numpy()[0]

    def visualize(self):
        total_reward = 0.0
        state = self.vis_env.reset()[0]
        done = False

        all_act = 0
        left_n, right_n = 0, 0

        while not done:
            action = self.get_action(state)
            all_act += 1
            if action == 1:
                right_n += 1
            if action == 0:
                left_n += 1
            next_state, reward, done, truncated, info = self.vis_env.step(action)
            total_reward += reward

            state = next_state

        print(f'Total reward of this displayed run is: {total_reward}')
        print(f'Fraction of rights: {right_n/all_act}')
        print(f'Fraction of lefts: {left_n/all_act}')

    def collect_trajectory(self, greedy: bool = False, record_video: bool = False):
        total_reward = 0.0
        states = []
        actions = []
        state = self.env.reset()[0]
        done = False
        frames = []

        while not done:
            if greedy:
                action = self.get_greedy_action(state)
            else:
                action = self.get_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            # adding to memory
            states.append(state)
            actions.append(action)
            state = next_state

            if record_video:
                camera_view = self.env.viewer.read_pixels()
                frames.append(camera_view)

        weights = [total_reward] * len(states)
        self.batch_states += states
        self.batch_actions += actions
        self.batch_weights += weights

        if greedy:
            print(f'Trajectory score: {total_reward}')

        if record_video:
            self.save_video(frames, name=f'{datetime.now()}')

    def update(self):
        n = len(self.batch_states)
        batch_states = torch.tensor(np.array(self.batch_states))
        batch_actions = torch.tensor(np.array(self.batch_actions))
        batch_weights = torch.tensor(np.array(self.batch_weights, dtype=float))

        self.optimizer.zero_grad()
        loss = -self.policy(batch_states).log_prob(batch_actions)*batch_weights
        loss = loss.sum() / n
        loss.backward()
        self.optimizer.step()
        score = sum(self.batch_weights)/len(self.batch_weights)

        print('------------------------------------------------------------------------------------------')
        # print(loss.item())
        print(f'Avg time of finishing: {len(self.batch_states)/self.sample_size}')
        trajectories_scores = torch.unique(batch_weights)
        print(f'Standard deviation of trajectory score: {torch.std(trajectories_scores).detach()}')
        print(f'Min, max trajectories: {min(trajectories_scores)}, {max(trajectories_scores)}')

        return score

    def save_video(self, frames, name, framerate=30, normal_dir: bool = True):
        import matplotlib
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        from IPython.display import HTML

        # Font sizes
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        height, width, _ = frames[0].shape
        dpi = 70
        orig_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
        fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
        matplotlib.use(orig_backend)  # Switch back to the original backend.
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])
        im = ax.imshow(frames[0])

        def update(frame):
            im.set_data(frame)
            return [im]

        interval = 1000 / framerate
        anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                       interval=interval, blit=True, repeat=False)

        html_video = HTML(anim.to_html5_video())
        if normal_dir:
            with open(f"videos/{name}.html", "w") as file:
                file.write(html_video.data)
        else:
            with open(f"{name}.html", "w") as file:
                file.write(html_video.data)

    def plot_policy_map(self):
        import numpy as np
        import matplotlib.pyplot as plt
        pos_left, pos_right = self.env.left_bound_xpos, self.env.wall_pos
        vel_left, vel_right = -2, 2
        grid_size = 0.1
        pos_grid, vel_grid = np.arange(pos_left, pos_right, grid_size), np.arange(vel_left, vel_right, grid_size)
        vx, vy = np.meshgrid(pos_grid, vel_grid, indexing='ij')
        f_val = np.zeros(shape=vx.shape)
        for i in range(vx.shape[0]):
            for j in range(vx.shape[1]):
                f_val[i][j] = self.policy(torch.tensor(np.asarray([vx[i][j], vy[i][j]]))).probs[0].item()

        plt.figure(figsize=(26, 25))
        plt.xticks(range(vx.shape[1]), np.round(vel_grid, 2), rotation=90)
        plt.yticks(range(vx.shape[0]), np.round(pos_grid, 2))
        plt.xlabel('velocity')
        plt.ylabel('position')
        plt.imshow(f_val, cmap='Blues')
        plt.show()

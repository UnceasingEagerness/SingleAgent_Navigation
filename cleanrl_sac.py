# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from buffers import ReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True  #This ensures the DL model gives same reproducible results when initialised with same random seed.
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SurfaceVessel-v0"
    """the environment id of the task"""
    total_timesteps: int =  300000
    """total timesteps of the experiments"""
    num_envs: int = 2
    """the number of parallel game environments"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005          #Used for soft update of the Value target network - To avoid the problem of moving target
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5000#5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4 #1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    checkpoint_freq: int = 20000
    """the frequency (in global steps) to save the full checkpoint"""
    surgical_resume: bool = False
    """whether to load the surgical 2D weights to kick off Phase 3"""
    checkpoint_path: str = None
    """path to a full_checkpoint.pth to resume training from"""
    goal_radius: float = 4.0
    """radius to consider goal reached"""
    obstacle_spacing: float = 50.0
    """spacing between obstacles"""
    obstacle_jitter: float = 5.0
    """The random jitter to shatter the grid into a minefield"""
    moving_target: bool = False
    """Whether the target is moving (Pursuit Evasion Phase 3)"""
    spawn_distance_min: float = 80.0
    """Minimum spawn distance for the goal"""
    spawn_distance_max: float = 150.0
    """Maximum spawn distance for the goal"""


from surfacevessel_env import SurfaceVesselEnv

def make_env(seed, idx, goal_radius, obstacle_spacing, obstacle_jitter, moving_target):
    def thunk():
        env = SurfaceVesselEnv(
            max_steps=10000, 
            show_viewport=False, 
            goal_radius=goal_radius,
            obstacle_spacing=args.obstacle_spacing,
            obstacle_jitter=args.obstacle_jitter,
            moving_target=args.moving_target,
            spawn_distance_range=(args.spawn_distance_min, args.spawn_distance_max)
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk


# ALGO LOGIC: initialize agent here:Dual branch which processes kinematics and lidar separately, then fuses them in the later layers. This allows the network to learn specialized features from each modality before combining them for decision making.
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()  #This initialises the base framework
        '''
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        ) #nn.Linear(inFeatures, outFeatures) where we pass in the state-action pair
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        '''
        # Branch A: Kinematics (6 inputs: dist, angle_yaw, speed, vel_x, vel_y, yaw_rate)
        self.kin_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Branch B: Perception (64 inputs: Normalized LiDAR beams)
        self.lidar_net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion Layer: 64 (from A) + 64 (from B) = 128 features + 2 Actions
        action_dim = np.prod(env.single_action_space.shape)
        self.q_net = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, x, a):
        # 1. Split the 70-dim observation
        kinematics = x[:, :6]
        lidar = x[:, 6:]
        
        # 2. Extract features independently
        kin_feat = self.kin_net(kinematics)
        lidar_feat = self.lidar_net(lidar)
        
        # 3. Fuse features and concatenate with the action
        fusion = torch.cat([kin_feat, lidar_feat], dim=1)
        q_in = torch.cat([fusion, a], dim=1)
        
        return self.q_net(q_in)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()     #Used whenever we derive our class from the base class
        # Branch A: Kinematics
        self.kin_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Branch B: Perception
        self.lidar_net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion Decision Layers
        self.decision_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        action_dim = np.prod(env.single_action_space.shape)
        self.fc_mean = nn.Linear(128, action_dim)
        self.fc_logstd = nn.Linear(128, action_dim)
        
        # Action rescaling buffers (CleanRL standard)
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        # 1. Split the 70-dim observation
        kinematics = x[:, :6]
        lidar = x[:, 6:]
        
        # 2. Extract features independently
        kin_feat = self.kin_net(kinematics)
        lidar_feat = self.lidar_net(lidar)
        
        # 3. Fuse and pass through decision network
        fusion = torch.cat([kin_feat, lidar_feat], dim=1)
        hidden = self.decision_net(fusion)
        
        # 4. Output Mean and Log Standard Deviation
        mean = self.fc_mean(hidden)
        log_std = self.fc_logstd(hidden)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample() 
        y_t = torch.tanh(x_t)
        
        # Scale to environment action limits
        action = y_t * self.action_scale + self.action_bias
        
        # Enforcing Action Bound (CleanRL specific math for SAC)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    import csv
    csv_file = open(f"runs/{run_name}/reward_components.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "r_total", "r_target", "step_penalty", "r_terminal", "dist_to_goal", "speed"])

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"PyTorch is currently using: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed, i, args.goal_radius, args.obstacle_spacing, args.obstacle_jitter, args.moving_target) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    if args.surgical_resume:
        print("Loading surgically grafted 2D weights for Phase 3!")
        actor.load_state_dict(torch.load("actor_2d_init.pth", map_location=device, weights_only=True))
        qf1.load_state_dict(torch.load("qf1_2d_init.pth", map_location=device, weights_only=True))
        qf2.load_state_dict(torch.load("qf2_2d_init.pth", map_location=device, weights_only=True))
        qf1_target.load_state_dict(torch.load("qf1_2d_init.pth", map_location=device, weights_only=True))
        qf2_target.load_state_dict(torch.load("qf2_2d_init.pth", map_location=device, weights_only=True))

    # PyTorch 2.0+ Compiler for massive speedup
    if int(torch.__version__.split('.')[0]) >= 2:
        actor = torch.compile(actor)
        qf1 = torch.compile(qf1)
        qf2 = torch.compile(qf2)
        qf1_target = torch.compile(qf1_target)
        qf2_target = torch.compile(qf2_target)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.checkpoint_path is not None:
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # If it's a full checkpoint (has 'actor' key)
        if 'actor' in checkpoint:
            actor.load_state_dict(checkpoint['actor'])
            if 'qf1' in checkpoint:
                qf1.load_state_dict(checkpoint['qf1'])
                qf2.load_state_dict(checkpoint['qf2'])
                qf1_target.load_state_dict(checkpoint['qf1_target'])
                qf2_target.load_state_dict(checkpoint['qf2_target'])
                q_optimizer.load_state_dict(checkpoint['q_optimizer'])
                actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                print("Full Checkpoint successfully loaded. Resuming Curriculum Phase!")
            else:
                print("Loaded Actor-only from dictionary.")
        else:
            # If it's an old-style checkpoint that was literally just the actor state_dict directly
            actor.load_state_dict(checkpoint)
            print("Loaded Historic Actor-only weights successfully! Q-networks will train from scratch.")

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        
        # FIX: If we load an expert checkpoint, do NOT reset the exploration noise to 1.0!
        if args.checkpoint_path is not None:
            log_alpha = torch.tensor([-3.0], requires_grad=True, device=device) # alpha ≈ 0.05
        else:
            log_alpha = torch.zeros(1, requires_grad=True, device=device) # alpha = 1.0
            
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    ep_count = 0
    best_return = -np.inf

    

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts and args.checkpoint_path is None:     #If the learning is in its early phases then it samples random actions.
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        r_target = infos.get("r_target", [0.0])[0]
        step_penalty = infos.get("step_penalty", [0.0])[0]
        r_terminal_info = infos.get("r_terminal", [0.0])[0]
        dist_t = infos.get("dist_to_goal", [0.0])[0]
        speed_t = infos.get("speed", [0.0])[0]
        r_term = float(r_terminal_info)
        csv_writer.writerow([global_step, rewards[0], r_target, step_penalty, r_term, dist_t, speed_t])
        
        if global_step % 100 == 0:
            csv_file.flush()
            writer.add_scalar("charts/r_target", r_target, global_step)
            writer.add_scalar("charts/step_penalty", step_penalty, global_step)

        #ep_count = 0

        '''
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_count += 1
                    ep_return = info['episode']['r'][0]
                    ep_len = info['episode']['l'][0]
                    dist = info.get('dist_to_goal', 0.0)[0]
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break
        '''
        # Gymnasium 1.0+ compatibility for episodic returns
        if "episode" in infos and "_episode" in infos:
            for idx, done in enumerate(infos["_episode"]):
                if done:
                    ep_count += 1
                    ep_return = float(infos["episode"]["r"][idx])
                    ep_len = int(infos["episode"]["l"][idx])
                    dist = float(dist_t)  # Approximation of final distance
                    
                    if ep_return > best_return:
                        best_return = ep_return
                        full_ckpt = {
                            'actor': actor.state_dict(),
                            'qf1': qf1.state_dict(),
                            'qf2': qf2.state_dict(),
                            'qf1_target': qf1_target.state_dict(),
                            'qf2_target': qf2_target.state_dict(),
                            'q_optimizer': q_optimizer.state_dict(),
                            'actor_optimizer': actor_optimizer.state_dict(),
                        }
                        torch.save(full_ckpt, f"runs/{run_name}/full_checkpoint_best.pth")
                        torch.save(actor.state_dict(), f"runs/{run_name}/actor_best.pth")
                    
                    if ep_count % 100 == 0:
                        torch.save(actor.state_dict(), f"runs/{run_name}/actor_ep{ep_count}.pth")
                        print(f"Checkpoint saved at EP {ep_count}")
                    
                    print(f"EP {ep_count} | env={idx} | step={global_step} | return={ep_return:.2f} | len={ep_len} | dist={dist:.2f} | best={best_return:.2f}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len, global_step)
                    writer.add_scalar("charts/dist_to_goal_final", dist, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        '''
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        '''
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                # AsyncVectorEnv stores final observations in a tuple or list at infos["final_observation"]
                if "final_observation" in infos and infos["final_observation"][idx] is not None:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)   #We sample from the replay buffer
            with torch.no_grad():
                next_state_actions, next_state_log_pi = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                if global_step % 50000 == 0 and global_step > 0:
                    torch.save(actor.state_dict(), f"runs/{run_name}/actor_{global_step}.pth")

            

    envs.close()
    writer.close()
    csv_file.close()

    # Save the actor and final full checkpoint
    final_ckpt = {
        'actor': actor.state_dict(),
        'qf1': qf1.state_dict(),
        'qf2': qf2.state_dict(),
        'qf1_target': qf1_target.state_dict(),
        'qf2_target': qf2_target.state_dict(),
        'q_optimizer': q_optimizer.state_dict(),
        'actor_optimizer': actor_optimizer.state_dict(),
    }
    torch.save(final_ckpt, f"runs/{run_name}/full_checkpoint_final.pth")
    torch.save(actor.state_dict(), f"runs/{run_name}/actor.pth")
    rb.save(f"runs/{run_name}/replay_buffer.npz")
    print(f"Model saved to runs/{run_name}/actor.pth")        
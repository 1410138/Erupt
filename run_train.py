import argparse
import os
import math
import logging
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from env import UdfJoinEnv
from executor import Executor
from cache import PlanCache
from dqn import DQNAgent, UdfJoinState
from ppo import PPOAgent
from config import DATASET_DIR, CACHE_DIR, MODEL_DIR, TIMEOUT, SAMPLE_MODE, SAMPLE_PERCENT, SAMPLE_ROWS
from schema import SchemaProvider
from selectivity import SelectivityProvider
from udf_features import UdfFeatureProvider
from udf_placement import TableNode, FilterNode, JoinNode
from cost_model import CostModel
from datetime import datetime

class ZeroUdfFeatureProvider(UdfFeatureProvider):
    """
    UdfFeatureProvider variant with all-zero UDF features.
    """
    def embed(self, t, expr, device: str = "cpu") -> torch.Tensor:
        # Return an all-zero vector with the original provider's dimension (128).
        return torch.zeros(self.d, dtype=torch.float32, device=device)

def run_episode(sql: str, agent: DQNAgent, executor: Executor, cache: PlanCache, greedy: bool = False, writer=None, episode_num=None, cost_model: CostModel = None):
    env = UdfJoinEnv(sql)
    total_reward = 0.0
    steps = 0
    episode_losses = []

    while (not env.is_terminal()):
        state = UdfJoinState.from_env(env)
        action = agent.choose_action(env, greedy=greedy)
        if action is None:
            break
        i, j, kind = action
        env.take_action(i, j, kind)
        steps += 1

        done = env.is_terminal()
        if done:
            if cost_model:
                plan = env.components[0].plan
                _, cost = cost_model.estimate_plan(plan, env.parsed)
                # Reward function: -log(cost).
                reward = -math.log(cost)
                ratio = 114514 # Placeholder.
            else:
                shape = env.current_shape()
                cached = cache.load_plan(sql, shape)
                if cached:
                    t_sec = float(cached.get("execution_time"))
                    final_sql = cached.get("rewritten_sql")
                else:
                    final_sql = env.build_final_sql()
                    t_sec, _ = executor.explain_analyze_time(final_sql)
                    cache.save_plan(sql, shape, final_sql, t_sec)
                baseline_time = executor.get_baseline_time(sql)
                
                # Smoothed reward.
                if t_sec == TIMEOUT:
                    ratio = math.exp(10)
                else:
                    ratio = t_sec / baseline_time
                reward = float(-math.log(ratio))
        else:
            reward = 0.0

        next_state = UdfJoinState.from_env(env)
        if not greedy:
            agent.remember(state, action, reward, next_state, float(done))
            loss = agent.optimize()
            if loss is not None:
                episode_losses.append(loss)

        total_reward += reward
    
    # Log to TensorBoard.
    if writer and episode_num and not greedy:
        writer.add_scalar('Training/Episode_Reward', total_reward, episode_num)
        writer.add_scalar('Training/Episode_Steps', steps, episode_num)
        writer.add_scalar('Training/Epsilon', agent.epsilon(), episode_num)
        writer.add_scalar('Training/Learning_Rate', agent.optimizer.param_groups[0]['lr'], episode_num)
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            writer.add_scalar('Training/Episode_Avg_Loss', avg_loss, episode_num)
    
    return total_reward, steps, ratio

def run_episode_ppo(sql: str, agent: PPOAgent, executor: Executor, cache: PlanCache,
                   greedy: bool = False, writer=None, episode_num=None, cost_model: CostModel = None):
    env = UdfJoinEnv(sql)
    total_reward = 0.0
    steps = 0
    ratio = 1.0

    while not env.is_terminal():
        action, log_prob, value, state = agent.choose_action(env, greedy=greedy)
        if action is None:
            break
        i, j, kind = action
        env.take_action(i, j, kind)
        steps += 1

        done = env.is_terminal()
        if done:
            if cost_model:
                plan = env.components[0].plan
                _, cost = cost_model.estimate_plan(plan, env.parsed)
                # Use -log(cost).
                reward = -math.log(cost)
                ratio = 114514
            else:
                t_sec = -float('inf')
                shape = env.current_shape()
                cached = cache.load_plan(sql, shape)
                if cached:
                    t_sec = float(cached.get("execution_time"))
                    final_sql = cached.get("rewritten_sql")
                else:
                    final_sql = env.build_final_sql()
                    t_sec, _ = executor.explain_analyze_time(final_sql)
                    cache.save_plan(sql, shape, final_sql, t_sec)
                baseline_time = executor.get_baseline_time(sql)

                if t_sec == TIMEOUT:
                    ratio = math.exp(10)
                else:
                    ratio = t_sec / baseline_time
                reward = float(-math.log(ratio))
        else:
            reward = 0.0

        if not greedy:
            agent.store_transition(state, action, log_prob, value, reward, float(done))

        total_reward += reward

    # Update PPO after accumulating a batch of recent experience.
    if writer and episode_num and not greedy:
        writer.add_scalar('Training/Episode_Reward', total_reward, episode_num)
        writer.add_scalar('Training/Episode_Steps', steps, episode_num)

    return total_reward, steps, ratio, None

def evaluate_val(sqls, agent: DQNAgent, executor: Executor, cache: PlanCache, writer=None, episode_num=None, cost_model: CostModel = None):
    ratios = []
    
    for sql in sqls:
        R, S, ratio = run_episode(sql, agent, executor, cache, greedy=True, cost_model=cost_model)
        ratios.append(ratio)
    
    gmrl = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    
    if writer and episode_num:
        writer.add_scalar('Validation/gmrl', gmrl, episode_num)
    
    return gmrl

def evaluate_val_ppo(sqls, agent: PPOAgent, executor: Executor, cache: PlanCache, writer=None, episode_num=None, cost_model: CostModel = None):
    ratios = []
    
    for sql in sqls:
        R, S, ratio, _ = run_episode_ppo(sql, agent, executor, cache, greedy=True, cost_model=cost_model)
        ratios.append(ratio)
    
    gmrl = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    
    if writer and episode_num:
        writer.add_scalar('Validation/gmrl', gmrl, episode_num)
    
    return gmrl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default="imdb")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--use_cost_model", type=bool, default=False)
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    args = parser.parse_args()

    CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if CUDA else "cpu")

    # Create the TensorBoard writer.
    log_dir = os.path.join(args.log_dir, f"{args.db_name}_train_{datetime.now().strftime('%m%d_%H%M%S')}_{args.algo}")
    if args.use_cost_model:
        log_dir += "_cost_model"
    writer = SummaryWriter(log_dir=log_dir)

    logging.basicConfig(filename=f'train_{args.db_name}_{args.algo}.log', level=logging.INFO, format='%(message)s')

    db_name = args.db_name
    with open(os.path.join(DATASET_DIR, db_name, f"train.sql"), "r") as f:
        train_sqls = [s for s in f.read().strip().split("\n") if s]
    with open(os.path.join(DATASET_DIR, db_name, f"val.sql"), "r") as f:
        val_sqls = [s for s in f.read().strip().split("\n") if s]

    # Initialize schema and selectivity providers.
    schema = SchemaProvider.from_file(db_name)
    selectivity = SelectivityProvider(db_name=db_name)

    if SAMPLE_MODE == "rows":
        suffix = str(SAMPLE_ROWS)
    else:
        suffix = str(SAMPLE_PERCENT)
    udf_feature_file = os.path.join(DATASET_DIR, db_name, f"udf_features_{suffix}.json")

    udf_provider = UdfFeatureProvider(udf_feature_file, d_model=128)
    # udf_provider = ZeroUdfFeatureProvider(udf_feature_file, d_model=128)
    
    if args.algo == "ppo":
        agent = PPOAgent(
            device=device,
            schema=schema,
            selectivity=selectivity,
            udf_provider=udf_provider,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            train_epochs=4,
            batch_size=64,
            max_grad_norm=1.0,
        )
    else:
        agent = DQNAgent(device=device, schema=schema, selectivity=selectivity, udf_provider=udf_provider)
    executor = Executor(db_name, timeout_sec=TIMEOUT)
    cache = PlanCache(root_dir=CACHE_DIR, db_name=db_name)

    cost_model = None
    if args.use_cost_model:
        cost_model = CostModel(
            table_node_cls=TableNode,
            filter_node_cls=FilterNode,
            join_node_cls=JoinNode,
            debug=False
        )
    
    # Log model configuration and hyperparameters.
    writer.add_text('Config/Database', db_name)
    writer.add_text('Config/Device', str(device))
    writer.add_text('Config/Episodes', str(args.episodes))
    writer.add_text('Config/Eval_Every', str(args.eval_every))
    writer.add_text('Config/Use_Cost_Model', str(args.use_cost_model))
    writer.add_text('Config/Algo', args.algo)
    
    # Log algorithm hyperparameters.
    if args.algo == "ppo":
        hyperparams = {
            'learning_rate': agent.optimizer.param_groups[0]['lr'],
            'gamma': agent.gamma,
            'gae_lambda': agent.gae_lambda,
            'clip_eps': agent.clip_eps,
            'entropy_coef': agent.entropy_coef,
            'value_coef': agent.value_coef,
            'train_epochs': agent.train_epochs,
            'batch_size': agent.batch_size,
            'max_grad_norm': agent.max_grad_norm,
        }
    else:
        hyperparams = {
            'learning_rate': agent.optimizer.param_groups[0]['lr'],
            'gamma': agent.gamma,
            'epsilon_start': agent.eps_start,
            'epsilon_end': agent.eps_end,
            'epsilon_decay': agent.eps_decay,
            'batch_size': agent.batch_size,
            'scheduler_step_size': 1000,
            'scheduler_gamma': 0.9,
        }
    for key, value in hyperparams.items():
        writer.add_text(f'Hyperparams/{key}', str(value))

    best_val_gmrl = 1e18
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    save_prefix = f"{timestamp}_{args.algo}"
    save_path_val = os.path.join(MODEL_DIR, db_name, f"{save_prefix}_best_policy_val.pt")
    os.makedirs(os.path.dirname(save_path_val), exist_ok=True)
    
    train_fn = run_episode if args.algo == "dqn" else run_episode_ppo
    eval_val_fn = evaluate_val if args.algo == "dqn" else evaluate_val_ppo
    
    # PPO update-control parameters.
    ppo_update_steps = 128  # Number of steps to accumulate before each update.
    current_steps_buffer = 0

    for i_episode in range(1, args.episodes + 1):
        sql = random.sample(train_sqls, 1)[0]
        if args.algo == "ppo":
            R, steps, _, _ = train_fn(sql, agent, executor, cache, greedy=False, writer=writer, episode_num=i_episode, cost_model=cost_model)
            
            # Accumulated PPO update.
            current_steps_buffer += steps
            if current_steps_buffer >= ppo_update_steps:
                update_info = agent.update()
                current_steps_buffer = 0
                
                # Log update losses.
                if update_info and writer:
                    writer.add_scalar('PPO/Policy_Loss', update_info.get('policy_loss', 0.0), i_episode)
                    writer.add_scalar('PPO/Value_Loss', update_info.get('value_loss', 0.0), i_episode)
                    writer.add_scalar('PPO/Entropy', update_info.get('entropy', 0.0), i_episode)
                    writer.add_scalar('PPO/Total_Loss', update_info.get('total_loss', 0.0), i_episode)
        else:
            R, steps, _ = train_fn(sql, agent, executor, cache, greedy=False, writer=writer, episode_num=i_episode, cost_model=cost_model)
            update_info = None
            
        logging.info(f"[Train ep {i_episode:05d}] reward={R:.4f} steps={steps}")

        if (i_episode % args.eval_every) == 0:
            gmrl = eval_val_fn(val_sqls, agent, executor, cache, writer=writer, episode_num=i_episode)
            logging.info(f"[Validate] gmrl={gmrl:.4f}")
            if gmrl < best_val_gmrl:
                best_val_gmrl = gmrl
                torch.save(agent.policy.state_dict(), save_path_val)
                logging.info(f"Saved best checkpoint to {save_path_val} (gmrl={gmrl:.4f})")
                writer.add_scalar('Training/Best_Validation_gmrl', gmrl, i_episode)
            
        if i_episode % 1000 == 0:
            checkpoint_path = os.path.join(MODEL_DIR, db_name, f"checkpoint_{i_episode:06d}.pt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(agent.policy.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

    writer.close()

if __name__ == "__main__":
    main()

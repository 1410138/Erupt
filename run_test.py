import argparse
import os
import logging
import torch
from env import UdfJoinEnv
from executor import Executor
from cache import PlanCache
from dqn import DQNAgent
from ppo import PPOAgent
from config import DATASET_DIR, CACHE_DIR, MODEL_DIR, TIMEOUT, SAMPLE_MODE, SAMPLE_ROWS, SAMPLE_PERCENT
from schema import SchemaProvider
from selectivity import SelectivityProvider
from udf_features import UdfFeatureProvider

class ZeroUdfFeatureProvider(UdfFeatureProvider):
    """
    UdfFeatureProvider variant with all-zero UDF features.
    """
    def embed(self, t, expr, device: str = "cpu") -> torch.Tensor:
        # Return an all-zero vector with the original provider's dimension (128).
        return torch.zeros(self.d, dtype=torch.float32, device=device)


def run_episode(sql: str, agent: DQNAgent, executor: Executor, cache: PlanCache):
    env = UdfJoinEnv(sql)
    total_reward = 0.0
    steps = 0

    while (not env.is_terminal()):
        action = agent.choose_action(env, greedy=True)
        if action is None:
            break
        i, j, kind = action
        env.take_action(i, j, kind)
        steps += 1

        done = env.is_terminal()
        if done:
            shape = env.current_shape()
            cached = cache.load_plan(sql, shape)
            if cached:
                t_sec = float(cached.get("execution_time"))
                final_sql = cached.get("rewritten_sql")
            else:
                final_sql = env.build_final_sql()
                t_sec, _ = executor.explain_analyze_time(final_sql)
                cache.save_plan(sql, shape, final_sql, t_sec) 
            reward = -t_sec
        else:
            reward = 0.0

        total_reward += reward
    
    return total_reward, steps

def run_episode_ppo(sql: str, agent: PPOAgent, executor: Executor, cache: PlanCache):
    env = UdfJoinEnv(sql)
    total_reward = 0.0
    steps = 0

    while not env.is_terminal():
        action, _, _, _ = agent.choose_action(env, greedy=True)
        if action is None:
            break
        i, j, kind = action
        env.take_action(i, j, kind)
        steps += 1

        done = env.is_terminal()
        if done:
            shape = env.current_shape()
            cached = cache.load_plan(sql, shape)
            if cached:
                t_sec = float(cached.get("execution_time"))
                final_sql = cached.get("rewritten_sql")
            else:
                final_sql = env.build_final_sql()
                t_sec, _ = executor.explain_analyze_time(final_sql)
                cache.save_plan(sql, shape, final_sql, t_sec) 
            reward = -t_sec
        else:
            reward = 0.0

        total_reward += reward
    
    return total_reward, steps

def evaluate(sqls, agent, executor: Executor, cache: PlanCache, algo: str = "dqn"):
    totals, baseline_totals = 0.0, 0.0
    execution_times = []
    baseline_times = []
    speedups = []
    
    for i, sql in enumerate(sqls):
        if algo == "ppo":
            R, S = run_episode_ppo(sql, agent, executor, cache)
        else:
            R, S = run_episode(sql, agent, executor, cache)

        baseline_time = executor.get_baseline_time(sql)
        execution_time = -R
        speedup = baseline_time / execution_time
        
        logging.info(f"[Test {i:05d}] baseline_time={baseline_time:.4f} execution_time={execution_time:.4f} speedup={speedup:.2f}x steps={S}")
        
        totals += execution_time
        baseline_totals += baseline_time
        execution_times.append(execution_time)
        baseline_times.append(baseline_time)
        speedups.append(speedup)
    
    return totals, baseline_totals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default="imdb")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--model_path", type=str, default='imdb_val_best.pt')
    args = parser.parse_args()

    CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if CUDA else "cpu")

    logging.basicConfig(filename=f'test_{args.db_name}.log', level=logging.INFO, format='%(message)s')

    db_name = args.db_name
    with open(os.path.join(DATASET_DIR, db_name, f"test.sql"), "r") as f:
        test_sqls = [s for s in f.read().strip().split("\n") if s]

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
        )
    else:
        agent = DQNAgent(device=device, schema=schema, selectivity=selectivity, udf_provider=udf_provider)
    executor = Executor(db_name, timeout_sec=TIMEOUT)
    cache = PlanCache(root_dir=CACHE_DIR, db_name=db_name)

    model_path = os.path.join(MODEL_DIR, db_name, args.model_path)
    state_dict = torch.load(model_path, map_location=device)
    agent.policy.load_state_dict(state_dict)
    agent.policy.eval()

    totals, baseline_totals = evaluate(test_sqls, agent, executor, cache, algo=args.algo)
    logging.info(f"[Test] totals={totals:.4f} baseline_totals={baseline_totals:.4f}")
    
    overall_speedup = baseline_totals / totals if totals > 0 else 1.0
    logging.info(f"[Test] overall_speedup={overall_speedup:.2f}x")

if __name__ == "__main__":
    main()

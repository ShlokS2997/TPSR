import json
import time
from symbolicregression.e2e_model import Transformer

from dyna_gym.agents.uct import UCT
from dyna_gym.agents.mcts import update_root, convert_to_json
from rl_env import RLEnv
from default_pi import E2EHeuristic
from sklearn.model_selection import KFold



def tpsr_fit(scaled_X, Y, params, equation_env, bag_number=1, rescale=True):
    # Prepare for cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=params.random_state)
    metrics_list = []

    # Iterate through each fold
    for train_index, val_index in kf.split(scaled_X[0]):
        x_to_fit = scaled_X[0][train_index]
        y_to_fit = Y[0][train_index]
        x_to_val = scaled_X[0][val_index]
        y_to_val = Y[0][val_index]

        samples = {'x_to_fit': 0, 'y_to_fit': 0, 'x_to_pred': 0, 'y_to_pred': 0}
        samples['x_to_fit'] = [x_to_fit]
        samples['y_to_fit'] = [y_to_fit]
        model = Transformer(params=params, env=equation_env, samples=samples)
        model.to(params.device)

        rl_env = RLEnv(
            params=params,
            samples=samples,
            equation_env=equation_env,
            model=model
        )

        dp = E2EHeuristic(
            equation_env=equation_env,
            rl_env=rl_env,
            model=model,
            k=params.width,
            num_beams=params.num_beams,
            horizon=params.horizon,
            device=params.device,
            use_seq_cache=not params.no_seq_cache,
            use_prefix_cache=not params.no_prefix_cache,
            length_penalty=params.beam_length_penalty,
            train_value_mode=params.train_value,
            debug=params.debug
        )

        start = time.time()

        agent = UCT(
            action_space=[],
            gamma=1., 
            ucb_constant=params.ucb_constant,
            horizon=params.horizon,
            rollouts=params.rollout,
            dp=dp,
            width=params.width,
            reuse_tree=True,
            alg=params.uct_alg,
            ucb_base=params.ucb_base
        )

        done = False
        s = rl_env.state
        ret_all = []
        for t in range(params.horizon):  # Use params.horizon instead of hard-coded value
            if len(s) >= params.horizon:
                print(f'Cannot process programs longer than {params.horizon}. Stop here.')
                break

            if done:
                break

            act = agent.act(rl_env, done)
            s, r, done, _ = rl_env.step(act)

            # Collect metrics here (e.g., store the reward)
            metrics_list.append(r)  # Example of collecting rewards

            update_root(agent, act, s)
            dp.update_cache(s)

        time_elapsed = time.time() - start

        # Evaluate the model on validation data
        y_val_pred = dp.get_predictions(x_to_val)  # Assuming you implement this method
        val_metrics = evaluate_metrics(y_to_val, None, y_val_pred)  # Implement this to evaluate your model
        metrics_list.append(val_metrics)

    # Return the aggregated metrics from all folds
    return metrics_list, time_elapsed, dp.sample_times

"""Task metrics for a MuJoCo locomotion policy.

`rlr test` reports the return, which for these tasks is dominated by a reward
term that changes meaning between task variants: with
`forward_reward_weight: -1.0` a good return means the agent travelled
*backwards*, and with a specification-shaped reward the number is in robustness
units and not comparable at all. This prints what the policy actually did.

Usage:
  python inspect_task.py <model.zip> [--episodes 10] [--seed 0]
                                [--forward-reward-weight -1.0] [--jump-height 1.5]
"""

import argparse

import gymnasium as gym
import numpy as np

import rlrom.utils as rlu


def rollout(model, env, episodes, seed=0, jump_height=1.5):
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        x0 = float(env.unwrapped.data.qpos[0])
        ret, vxs, zs = 0.0, [], []
        steps = 0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += float(reward)
            vxs.append(float(info["x_velocity"]))
            zs.append(float(obs[0]))
            steps += 1
            if terminated or truncated:
                break
        rows.append(dict(
            ret=ret, steps=steps,
            dx=float(env.unwrapped.data.qpos[0]) - x0,
            vx=float(np.mean(vxs)), abs_vx=float(np.mean(np.abs(vxs))),
            z=float(np.mean(zs)), z_max=float(np.max(zs)),
            frac_still=float(np.mean(np.abs(vxs) < 0.3)),
            frac_jump_in_place=float(np.mean(
                (np.array(zs) > jump_height) & (np.abs(vxs) < 0.3))),
        ))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--env-id", default="Hopper-v4")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forward-reward-weight", type=float, default=None,
                   help="pass -1.0 to match a backwards-trained run")
    p.add_argument("--jump-height", type=float, default=1.5,
                   help="z threshold considered a jump above standing height")
    args = p.parse_args()

    model = rlu.load_model(args.model)
    if model is None:
        raise SystemExit(f"could not load a policy from {args.model}")

    kwargs = {}
    if args.forward_reward_weight is not None:
        kwargs["forward_reward_weight"] = args.forward_reward_weight
    env = gym.make(args.env_id, **kwargs)

    rows = rollout(model, env, args.episodes, args.seed, args.jump_height)
    env.close()

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"{args.model}  ({args.episodes} episodes, env {args.env_id}"
          + (f", forward_reward_weight {args.forward_reward_weight}"
             if args.forward_reward_weight is not None else "") + ")")
    print(f"  return            {m('ret'):10.1f}")
    print(f"  episode length    {m('steps'):10.1f}  -> completed "
          f"{np.mean([r['steps'] >= 1000 for r in rows]) * 100:.0f}%")
    print(f"  x displacement    {m('dx'):10.2f} m   "
          f"({'backwards' if m('dx') < 0 else 'forwards'})")
    print(f"  mean v_x          {m('vx'):10.2f} m/s")
    print(f"  mean z            {m('z'):10.2f}      max z {m('z_max'):.2f}")
    print(f"  time |v_x| < 0.3  {m('frac_still') * 100:10.0f} %")
    print(f"  time z>{args.jump_height} and |v_x|<0.3  "
          f"{m('frac_jump_in_place') * 100:6.0f} %")


if __name__ == "__main__":
    main()

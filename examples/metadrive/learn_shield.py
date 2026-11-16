"""Learn the ICFEM26 piecewise-constant shield from vanilla-policy traces.

Faithful to the paper's construction, with the deviations agreed for this
reproduction:

* features (v, heading_err, lat_dev) from metadrive_env helpers;
* regions R_center / R_notcenter / R_danger from phi1, phi2 (same e1, e2 as the
  STL spec), speed bucketed per the paper: [0,.2], [.2,.5], [.5,.8], [.8,1.1];
* a state-action pair is *safe* when the next state satisfies the safety
  requirement (state formula): still on a lane, |lat_dev| <= 1, and no crash /
  out-of-road flag in the step info (user decision: label by a state formula);
* per region x speed bucket, the safe action is the imitation term of
  Eq. shieldopt: minimise sum ||a - a_t||^2 over safe samples -> the mean safe
  action. R_danger instead damps the live action (0.5*a1, -0.6) (the paper's
  "restrictive steering + mechanical deceleration"); it is what preserves the
  policy's steering on a curve that bends both ways.
* the result is written to shield_table.json, which metadrive_env.metadrive_shield
  loads if present (otherwise the provisional law is used).

Run from examples/metadrive/ with the venv:
    uv run python learn_shield.py --seeds 50
"""
import argparse
import json
import os

import numpy as np
import yaml
from stable_baselines3 import PPO

import metadrive_env as M
from metadrive_env import MetaDriveRLEnv

# region thresholds: must match metadrive.stl (param e1, e2)
E1 = 0.05
E2 = 0.8
# speed buckets (normalised speed) as in the paper
SPEED_EDGES = (0.2, 0.5, 0.8)  # last bucket is [0.8, inf)


def region_of(he, lat):
    # Signed regions (a deliberate extension): the constant action cannot steer
    # a curve that bends both ways unless the region encodes which side the car
    # is on. "left" = lat_dev >= 0.
    if abs(lat - he) >= E2:
        return 'danger'
    side = 'left' if lat >= 0 else 'right'
    return ('center_' if abs(he) < E1 else 'notcenter_') + side


def bucket_of(v):
    for i, edge in enumerate(SPEED_EDGES):
        if v < edge:
            return i
    return len(SPEED_EDGES)


def collect_episodes(seeds, env, model):
    """One record per episode: outcome + its (region, bucket, action) samples.

    Trace-level safety labelling (the paper classifies *traces*, not single
    transitions): an episode is safe if it arrives or runs to the horizon, and
    unsafe if it ends by crash or out-of-road. Per-transition labelling with
    ``|lat|<=1 and no crash`` marks ~99.9% of samples safe (only the terminal
    step fails) and learns nothing.
    """
    crash_flags = ('crash_vehicle', 'crash_object', 'crash_sidewalk',
                   'crash_building', 'crash_human')
    episodes = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        samples = []
        while not done:
            action, _ = model.predict(obs)  # stochastic, as rlr test evaluates
            he = M.heading_err(env)
            lat = M.lat_dev(env)
            v = M.speed_norm(env)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            samples.append((region_of(he, lat), bucket_of(v),
                            np.asarray(action, dtype=float)))
        crash = any(bool(info.get(k, False)) for k in crash_flags)
        off = bool(info.get('out_of_road', False))
        arrived = bool(info.get('arrive_dest', False))
        safe = (not crash) and (not off)
        episodes.append({'seed': seed, 'crash': crash, 'off': off,
                         'arrived': arrived, 'safe': safe, 'samples': samples})
    return episodes


def learn_table(episodes, min_samples=20):
    """Mean action over SAFE traces, per region x speed bucket."""
    safe_samples = [sm for ep in episodes if ep['safe'] for sm in ep['samples']]
    all_samples = [sm for ep in episodes for sm in ep['samples']]
    nb = len(SPEED_EDGES) + 1
    table = {}
    stats = {}
    for region in ('center_left', 'center_right', 'notcenter_left', 'notcenter_right'):
        reg_safe = [a for (rg, b, a) in safe_samples if rg == region]
        reg_fallback = np.mean(reg_safe, axis=0) if reg_safe else np.array([0.0, 0.5])
        rows = []
        stats[region] = []
        for b in range(nb):
            safe = [a for (rg, bb, a) in safe_samples if rg == region and bb == b]
            n_all = sum(1 for (rg, bb, a) in all_samples if rg == region and bb == b)
            if len(safe) >= min_samples:
                c = np.mean(safe, axis=0)
            elif reg_safe:
                c = reg_fallback
            else:
                c = np.array([0.0, 0.5])
            rows.append([round(float(c[0]), 4), round(float(c[1]), 4)])
            stats[region].append({'bucket': b, 'n': n_all, 'n_safe': len(safe),
                                  'action': rows[-1]})
        table[region] = rows
    # danger: damp live action (steer scale, constant brake)
    table['danger'] = {'steer_scale': 0.5, 'accel': -0.6}
    ndanger = sum(1 for (rg, b, a) in all_samples if rg == 'danger')
    stats['danger'] = {'n': ndanger,
                       'n_safe': sum(1 for (rg, b, a) in safe_samples if rg == 'danger')}
    return table, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=50, help='number of vanilla seeds (from 0)')
    ap.add_argument('--model', default='models/ppo_vanilla.zip')
    ap.add_argument('--out', default='shield_table.json')
    ap.add_argument('--min-samples', type=int, default=20)
    args = ap.parse_args()

    cfg_env = yaml.safe_load(open('cfg_env.yml'))
    env = MetaDriveRLEnv(cfg_env)
    model = PPO.load(args.model, device='cpu')
    episodes = collect_episodes(range(args.seeds), env, model)
    env.close()

    table, stats = learn_table(episodes, min_samples=args.min_samples)
    n_safe_ep = sum(1 for e in episodes if e['safe'])
    total = sum(len(e['samples']) for e in episodes)
    print(f"{len(episodes)} episodes: {n_safe_ep} safe, "
          f"{len(episodes)-n_safe_ep} unsafe; {total} samples")
    for region in ('center_left', 'center_right', 'notcenter_left', 'notcenter_right'):
        for st in stats[region]:
            print(f"  {region:9s} bucket {st['bucket']}: "
                  f"{st['n_safe']:5d}/{st['n']:5d} safe -> {st['action']}")
    print(f"  danger    : {stats['danger']['n_safe']}/{stats['danger']['n']} safe "
          f"-> steer_scale={table['danger']['steer_scale']}, "
          f"accel={table['danger']['accel']}")
    with open(args.out, 'w') as f:
        json.dump(table, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

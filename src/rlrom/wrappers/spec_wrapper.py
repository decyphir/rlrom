import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import stlrom
from rlrom.utils import append_to_field_array as add_metric
from rlrom.wrappers.reward_machine import RewardMachine, _validate_action_value
from gymnasium.wrappers import FlattenObservation

import importlib

def wrap_env_specs(env,cfg):
    
    cfg_specs = cfg.get('cfg_specs', None)            
    if cfg_specs is not None:
        env = SpecWrapper(env, cfg)
        
        if cfg_specs.get("flatten_obs", True):
            env = FlattenObservation(env)          
    
    return env

class SpecWrapper(gym.Wrapper): 
    
    def __init__(self,env,cfg):        
        
        gym.Wrapper.__init__(self, env)        
        self.env = env        
        self.cfg = cfg
        
        # Parsing and adding STL formulas
        stl_driver= stlrom.STLDriver()        
        cfg_specs = cfg.get('cfg_specs',{})
        stl_specs_str = cfg_specs.get('specs','') # keeping for backward compat
        stl_specs_str = cfg_specs.get('stl_specs',stl_specs_str) 
        
        if stl_specs_str=='':    # default stl signals declaration. Not sure why it is here.
            stl_specs_str = 'signal'
            first = True
            for a in cfg_specs.get('action_names',{}):
                if first:
                    stl_specs_str += ' '+ a
                    first = False
                else:
                    stl_specs_str += ','+ a
            for o in cfg_specs.get('obs_names',{}):
                if first:
                    stl_specs_str += ' '+ o
                    first = False
                else:
                    stl_specs_str += ','+ o
            stl_specs_str += ',env_reward'                                 
    
        stl_driver.parse_string(stl_specs_str)
        self.stl_specs_str = stl_specs_str
        self.real_time_step = cfg_specs.get('real_time_step',1)
        # 'monitor_formulas' is the unique formula section: every formula is
        # monitored once per step and always produces a *label* for the reward
        # machines (true when its lower robustness bound is > 0). A non-zero
        # 'obs_weight' additionally exposes it in the observation (scaled),
        # a non-zero 'reward_weight' additionally shapes the reward (scaled).
        # 'label_formulas' (legacy name) and the 'obs_formulas' /
        # 'reward_formulas' sections (legacy) are still accepted but deprecated.
        self.obs_formulas = cfg_specs.get('obs_formulas',{})
        self.reward_formulas = cfg_specs.get('reward_formulas',{})
        if 'obs_formulas' in cfg_specs or 'reward_formulas' in cfg_specs:
            warnings.warn(
                "cfg_specs 'obs_formulas'/'reward_formulas' are deprecated; "
                "declare the formulas once in 'monitor_formulas' and use "
                "'obs_weight'/'reward_weight' instead", DeprecationWarning, stacklevel=2)
        if 'label_formulas' in cfg_specs:
            if 'monitor_formulas' in cfg_specs:
                raise ValueError(
                    "cfg_specs: use 'monitor_formulas' only, not both "
                    "'monitor_formulas' and the legacy 'label_formulas'")
            warnings.warn(
                "cfg_specs 'label_formulas' is a legacy alias for "
                "'monitor_formulas'", DeprecationWarning, stacklevel=2)
            cfg_specs['monitor_formulas'] = cfg_specs['label_formulas']
        self.monitor_formulas = cfg_specs.get('monitor_formulas',{})
        # Effective obs/reward entries: (f_name, f_opt, res_f prefix).
        # Legacy sections keep their order and behavior first, then monitor
        # formulas with a non-zero weight are appended. Monitor formulas are
        # evaluated once per step (in the monitor pass of step()) and their
        # value is reused here, so a formula playing several roles is not
        # evaluated twice.
        self._obs_entries = [(n, o, 'obs_formulas') for n, o in self.obs_formulas.items()]
        self._reward_entries = [(n, o, 'reward_formulas') for n, o in self.reward_formulas.items()]
        for f_name, f_opt in self.monitor_formulas.items():
            f_opt = f_opt or {}
            obs_w = f_opt.get('obs_weight', 0)
            rew_w = f_opt.get('reward_weight', 0)
            if obs_w and f_name in self.obs_formulas:
                raise ValueError(
                    f"Formula '{f_name}' is declared in both 'obs_formulas' "
                    "(deprecated) and 'monitor_formulas' with a non-zero "
                    "'obs_weight'; keep it in one place only")
            if rew_w and f_name in self.reward_formulas:
                raise ValueError(
                    f"Formula '{f_name}' is declared in both 'reward_formulas' "
                    "(deprecated) and 'monitor_formulas' with a non-zero "
                    "'reward_weight'; keep it in one place only")
            if obs_w:
                self._obs_entries.append((f_name, f_opt, 'monitor_formulas'))
            if rew_w:
                self._reward_entries.append((f_name, f_opt, 'monitor_formulas'))
        # Optional per-formula action constraints: a monitor formula may carry
        # an 'actions' section (same fields as a reward-machine state: enable,
        # blocked, forced, fallback, priority). When the formula is true, its actions
        # are enforced on the agent's action from the NEXT step on (the truth
        # is only known after the env step, mirroring how RM states persist),
        # through the same conflict resolution as machine actions.
        self.monitor_actions = {}
        for f_name, f_opt in self.monitor_formulas.items():
            f_opt = f_opt or {}
            actions = f_opt.get('actions')
            if actions is None:
                continue
            if not isinstance(actions, dict):
                raise ValueError(
                    f"SpecWrapper: monitor formula '{f_name}' field 'actions' "
                    f"must be a dict (enable/blocked/forced/fallback), "
                    f"got {actions!r}")
            blocked = actions.get('blocked', [])
            if not isinstance(blocked, (list, tuple)):
                raise ValueError(
                    f"SpecWrapper: monitor formula '{f_name}' field "
                    f"'actions.blocked' must be a list of raw action values, "
                    f"got {blocked!r}")
            self.monitor_actions[f_name] = {
                'enable': bool(actions.get('enable', True)),
                'blocked': [_validate_action_value(v, 'actions.blocked', f_name)
                            for v in blocked],
                'forced': _validate_action_value(
                    actions.get('forced'), 'actions.forced', f_name),
                'fallback': _validate_action_value(
                    actions.get('fallback'), 'actions.fallback', f_name),
                'action_priority': actions.get('priority', 0),
            }
        # monitor formulas that were true at the previous step and declare
        # (enabled) actions; consumed by _apply_shield at the start of step()
        self._active_formula_actions = []
        self.env_reward_weight = cfg_specs.get('env_reward_weight', 1.0)
        self.multi_objective = cfg_specs.get('multi_objective',False)
        self.keep_old_reward = cfg_specs.get('keep_old_reward', False)
        self.eval_formulas = cfg_specs.get('eval_formulas',{})
        self._reward_machines_cfg = cfg_specs.get('reward_machines',[])
        self.debug_signals=  cfg_specs.get('debug_signals',False)
        self.debug_formulas= cfg_specs.get('debug_formulas',False)
        self.debug_rm= cfg_specs.get('debug_rm',False)
        self.debug_actions= cfg_specs.get('debug_actions',False)
        self.disable_actions= cfg_specs.get('disable_actions',False)
        self.BigM = cfg_specs.get('BigM')
        self.semantics= 'Boolean'  # TODO option to change. For now Boolean is simpler to interpret rewards
        self.stl_driver = stl_driver                        

        self.time_step = 0     # integer current time step 
        self.current_time = 0  # real time (=time_step*real_time_step)  for formula evaluation
        self._last_action = None  # last shielded action executed (shield fallback)
        self.episode={}
        to_import = cfg.get('import_module')                
        if to_import is not None:
            import_module = importlib.import_module(to_import)
        else:
            import_module = None

        self.import_module= import_module
        
        # define signals_map: map signal names to their expressions given in cfg_specs
        cfg_specs= cfg.get('cfg_specs')
        self.signals_map={}
        self._action_signal_names = set(cfg_specs.get('action_names', {}))
        if 'action_names' in cfg_specs:
            for a_name,a_ref in cfg_specs['action_names'].items():                                    
                self.signals_map[a_name]=a_ref            
        if 'obs_names' in cfg_specs:
            for o_name,o_ref in cfg_specs['obs_names'].items():                     
                self.signals_map[o_name]=o_ref
        if 'aux_sig_names' in cfg_specs: # TODO document/test/example this
            for o_name,o_ref in cfg_specs['aux_sig_names'].items():                     
                self.signals_map[o_name]=o_ref
        # The env (original) reward is sampled into the STL driver under the signal
        # name 'env_reward'. The value is the raw reward returned by the wrapped
        # environment, i.e. the `reward` variable in scope (there is no local named
        # 'env_reward'); get_sample evaluates this expression per step. The
        # agent-visible/shaped reward (the SpecWrapper output) is exposed
        # separately as 'reward' in the plotting / get_signal_from_str namespace.
        self.signals_map['env_reward'] = 'reward'

        # signals in specs
        self.signals_specs = stl_driver.get_signals_names().split()               
        self.signals_specs_idx = {}  # index map of signals in stl samples (t, s1, s2, etc).
        idx = 1
        for s in self.signals_specs:
            self.signals_specs_idx[s] = idx
            idx+=1

        num_obs_formulas = len(self._obs_entries)

        # Build the reward machines (FSMs) declared under cfg_specs.reward_machines.
        self.reward_machines = []
        for rm_cfg in self._reward_machines_cfg:
            rm_cfg = dict(rm_cfg)
            machine_cfg = rm_cfg.pop('reward_machine', None)
            if machine_cfg is None:
                machine_cfg = rm_cfg.pop('fsm', None)
           
            if machine_cfg is None:
                raise ValueError("reward_machines entry missing required 'reward_machine' or 'fsm' sub-config")
            self.reward_machines.append({
                'name': rm_cfg.pop('name', f'rm_{len(self.reward_machines)}'),
                'weight': rm_cfg.pop('weight', 1.0),
                'in_observation': rm_cfg.pop('in_observation', False),
                'action_priority': rm_cfg.pop('action_priority', 0),
                'machine': RewardMachine(machine_cfg),
                'cfg': rm_cfg,
            })
        if self.BigM is None:
            self.BigM = stlrom.Signal.get_BigM()

        # Build the (possibly dict) observation space: always has 'unwrapped';
        # adds 'obs_formulas' if obs formulas exist, and 'rm_states' if reward
        # machines exist. A dict-based obs supports any underlying obs space.
        num_rm = len(self.reward_machines)
        dict_fields = {'unwrapped': env.observation_space}
        if num_obs_formulas > 0:
            obs_formula_space = spaces.Box(
                np.array([-self.BigM]*num_obs_formulas),
                np.array([self.BigM]*num_obs_formulas))
            dict_fields['obs_formulas'] = obs_formula_space
        if num_rm > 0:
            rm_states_space = spaces.Box(
                low=0, high=num_rm, shape=(num_rm,), dtype=np.int_)
            dict_fields['rm_states'] = rm_states_space

        if len(dict_fields) == 1:
            self.observation_space = env.observation_space
        else:
            self.observation_space = spaces.Dict(dict_fields)

        if self.multi_objective:
            # See https://mo-gymnasium.farama.org/tutorials/custom_env/
            self.reward_dim = len(self._reward_entries)
            if self.keep_old_reward:
                self.reward_dim += 1
            self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))
            self.unwrapped.reward_dim = self.reward_dim
            self.unwrapped.reward_space = self.reward_space

    def reset(self, **kwargs):        
        self.time_step = 0
        self.current_time = 0
        self.episode={'observations':[], 'actions':[],'rewards':[], 'env_rewards':[],'dones':[], 'last_obs':[], 
                      'stl_data':[], 'labels':[], 'rm_states':[], 'rm_rewards':[]}        
        obs0, info = self.env.reset(**kwargs)
        self.wrapped_obs = obs0        
        robs0 = self.reset_monitor()
        self.reset_rm()
        self._last_action = None
        self._active_formula_actions = []
        rm_states0 = self._rm_states_obs()
        obs = self._build_obs(obs0, robs0, rm_states0)
        self.last_obs = obs
        self.episode['last_obs'] = obs

        # initialize res_f. It will store monitoring of formulas computed during the episode.
        # Keys are namespaced by group ('monitor_formulas/<f>', 'obs_formulas/<f>',
        # 'reward_formulas/<f>', ...): a formula may appear in several groups
        # (legacy sections), and a shared key would be appended twice per step.
        res_f={}
        for f_name, _ in self.obs_formulas.items():
            res_f[f'obs_formulas/{f_name}']=[]
        for f_name, _ in self.reward_formulas.items():
            res_f[f'reward_formulas/{f_name}']=[]
        for f_name, _ in self.monitor_formulas.items():
            res_f[f'monitor_formulas/{f_name}']=[]
        self.episode['res_f'] = res_f
        return obs, info

    def reset_rm(self):
        """Reset all reward machines to their initial state."""
        for rm in self.reward_machines:
            rm['machine'].reset()

    def _apply_shield(self, action):
        """Apply action constraints to the agent's action, from two sources:

        * reward machines: each machine's current state may declare an
          ``actions`` section (see ``RewardMachine``); a state whose
          ``actions.enable`` is False imposes nothing;
        * monitor formulas with an ``actions`` section that were *true* at
          the previous step (``self._active_formula_actions``).

        In both sources: ``actions.forced`` forces that raw action; 
        ``actions.blocked`` forbids the listed raw actions.

        Blocking is enforced cumulatively: every source's blocked list
        constrains the final action, and the replacement (previous action or
        fallback) must not be blocked by any source. The priority --
        ``action_priority`` for reward machines (machine-level),
        ``actions.priority`` for monitor formulas (default 0) -- is only
        consulted when sources conflict. Sources that *agree* never
        conflict, even at equal priority. A ValueError is raised when equal
        priorities cannot be resolved:
          * two sources force *different* actions at the same max priority
            (forcing the *same* action at a tie is fine);
          * a forced action is blocked by a source of *equal* priority
            (a blocker of strictly higher priority wins and provides the
            replacement; a lower-priority blocker is ignored);
          * several equal-max-priority blockers require a fallback and
            their ``fallback`` values differ (identical fallbacks are fine;
            a None fallback defers to the one that is set).
        When the candidate is blocked it is replaced by the previously
        executed action (when that one is not blocked by any source), else
        by the fallback. A fallback is trusted even if another source would
        also block it (it is the configured last resort).

        Returns ``(action, info)``; ``info`` is None when the action passes
        through unchanged, else a dict describing the override (also surfaced
        as ``info['shield']`` in the env step info; ``info['shield']['machine']
        is the name of the machine or monitor formula that triggered it).

        ``cfg_specs.disable_actions: True`` disables every action constraint
        (machine and monitor-formula) at once: the agent's action is always
        returned unchanged.
        """
        if self.disable_actions:
            return action, None
        forcers = []
        blockers = []   # (priority, name, fallback, blocked list)
        for rm in self.reward_machines:
            m = rm['machine']
            if not m.actions_enabled():
                continue
            forced = m.forced()
            if forced is not None:
                forcers.append((rm['action_priority'], rm['name'], forced))
            else:
                blocked = m.blocked()
                if blocked:
                    blockers.append(
                        (rm['action_priority'], rm['name'], m.fallback(), blocked))
        # monitor formulas that were true at the previous step and whose
        # actions are enabled act like a machine state constraint
        for f_name in self._active_formula_actions:
            a = self.monitor_actions[f_name]
            if not a['enable']:
                continue
            if a['forced'] is not None:
                forcers.append((a['action_priority'], f_name, a['forced']))
            elif a['blocked']:
                blockers.append(
                    (a['action_priority'], f_name, a['fallback'], a['blocked']))

        def tie_hint():
            return ("set distinct priorities ('action_priority' for "
                    "machines, 'actions.priority' for monitor formulas)")

        candidate = action
        forcer = None
        forcer_names = []
        if forcers:
            max_priority = max(f[0] for f in forcers)
            winners = [f for f in forcers if f[0] == max_priority]
            forced_actions = {w[2] for w in winners}
            if len(forced_actions) > 1:
                # a tie only conflicts when the forced actions differ
                raise ValueError(
                    "SpecWrapper: shield conflict -- sources "
                    f"{[w[1] for w in winners]} share max action priority "
                    f"{max_priority} but force different actions "
                    f"{sorted(forced_actions)}; {tie_hint()}")
            forcer = winners[0]
            forcer_names = [w[1] for w in winners]
            candidate = forcer[2]

        # blockers that object to a given action: their blocked list contains
        # it and (when a forcer is in play) they strictly outrank the forcer;
        # a blocker *tying* the forcer is a conflict, checked below
        def objectors(a):
            return [b for b in blockers
                    if a in b[3] and (forcer is None or b[0] > forcer[0])]

        if forcer is not None:
            tied = [b for b in blockers
                    if candidate in b[3] and b[0] == forcer[0]]
            if tied:
                raise ValueError(
                    "SpecWrapper: shield conflict -- forced action "
                    f"{candidate} by {forcer_names} (priority "
                    f"{forcer[0]}) is blocked at the same priority by "
                    f"{[t[1] for t in tied]}; {tie_hint()}")

        reason = 'forced' if forcer is not None else None
        name = ', '.join(forcer_names) if forcer is not None else None
        obj = objectors(candidate)
        if obj:
            max_priority = max(b[0] for b in obj)
            winners = [b for b in obj if b[0] == max_priority]
            name = ', '.join(w[1] for w in winners)
            prev = self._last_action
            if prev is not None and not objectors(prev):
                candidate = prev
                reason = 'blocked->previous'
            else:
                # all max-priority blockers block the candidate: their
                # fallbacks must agree (identical or None) to not conflict
                fallbacks = {w[2] for w in winners if w[2] is not None}
                if len(fallbacks) > 1:
                    raise ValueError(
                        "SpecWrapper: shield conflict -- sources "
                        f"{[w[1] for w in winners]} share max action "
                        f"priority {max_priority} and all block action "
                        f"{candidate}, but their 'fallback' actions "
                        f"differ {sorted(fallbacks)}; {tie_hint()}")
                fb = fallbacks.pop() if fallbacks else None
                if fb is None:
                    raise ValueError(
                        f"SpecWrapper: '{name}' blocks action {candidate}, but "
                        f"the previous action ({prev!r}) is also blocked and "
                        "no 'fallback' action is defined")
                candidate = fb
                reason = 'blocked->fallback'

        if candidate == action:
            return action, None
        return candidate, {
            'machine': name, 'reason': reason,
            'original': action, 'shielded': candidate,
        }

    def _rm_states_obs(self):
        """Current RM state indices (one per reward machine) as a numpy array."""
        return np.array([rm['machine'].state_idx for rm in self.reward_machines], dtype=np.int_)

    def _build_obs(self, unwrapped, robs, rm_states):
        """Assemble the (possibly dict) observation returned to the agent."""
        if not isinstance(self.observation_space, spaces.Dict):
            return unwrapped
        obs = {'unwrapped': unwrapped}
        if 'obs_formulas' in self.observation_space.spaces:
            obs['obs_formulas'] = robs
        if 'rm_states' in self.observation_space.spaces:
            obs['rm_states'] = rm_states
        return obs

    def show_reward_machines(self):
        """Print a readable, text summary of the configured reward machines.

        Intended as a lightweight visual/text diagnostic of the RM structure
        (states, transitions, priorities) and the label formulas that drive them.
        """
        if not self.reward_machines:
            print("No reward machines configured.")
            return
        if self.monitor_formulas:
            print("Monitor formulas:")
            for f_name, f_opt in self.monitor_formulas.items():
                print(f"  {f_name}: {f_opt}")
        for i, rm in enumerate(self.reward_machines):
            m = rm['machine']
            print(f"\nReward machine [{i}] '{rm['name']}' (weight={rm['weight']}"
                  f", action_priority={rm.get('action_priority', 0)})")
            print("  States:")
            for sid in m.states:
                flags = []
                if m.is_final_state(sid):
                    flags.append('final')
                if m._is_initial.get(sid):
                    flags.append('initial')
                tag = (' [' + ', '.join(flags) + ']') if flags else ''
                shields = []
                if not m.actions_enabled(sid):
                    shields.append('actions: disabled')
                elif m.blocked(sid):
                    shields.append(f"blocked={m.blocked(sid)}")
                if m.forced(sid) is not None:
                    shields.append(f"forced={m.forced(sid)}")
                if m.fallback(sid) is not None:
                    shields.append(f"fallback={m.fallback(sid)}")
                shield_tag = ('  [' + ', '.join(shields) + ']') if shields else ''
                print(f"    {sid}{tag}: reward={m.state_rewards[sid]} idx={m.states_idx_map[sid]}{shield_tag}")
            print("  Transitions:")
            for t in m.transitions:
                print(f"    {t['id']}: {t['from']} -> {t['to']} if '{t['condition']}'"
                      f" reward={t['reward']} priority={t['priority']}")

    def reset_monitor(self):        
        # recreate the driver to reset the data (stlrom has no dedicated reset API,
        # and re-parsing the same driver accumulates parse state)
        self.stl_driver = stlrom.STLDriver()
        self.stl_driver.parse_string(self.stl_specs_str)
        self._validate_formula_params()
        self.episode['stl_data'] = self.stl_driver.data
        return [0]*len(self._obs_entries) 

    def _validate_formula_params(self):
        """Check that parameters referenced by formula_params are declared in
        the spec string. stlrom parameters can only be created with a
        'param <name> = <value>' statement in the spec; get_param/set_param
        raise for undeclared names.
        """
        checked = set()
        for group in ('obs_formulas', 'reward_formulas', 'monitor_formulas',
                      'eval_formulas'):
            for f_name, f_opt in getattr(self, group).items():
                if not isinstance(f_opt, dict):
                    continue
                for p in (f_opt.get('formula_params') or {}):
                    if p in checked:
                        continue
                    checked.add(p)
                    try:
                        self.stl_driver.get_param(p)
                    except Exception:
                        raise ValueError(
                            f"formula_params of '{group}/{f_name}' references "
                            f"parameter '{p}', which is not declared in "
                            f"cfg_specs.specs. Declare it there, e.g.:\n"
                            f"  param {p} = <default_value>")

    def step(self, action):
        # shield: machines may replace the agent's action before it reaches the env
        action, shield_info = self._apply_shield(action)
        if self.debug_actions and shield_info is not None:
            print(f"t:{self.time_step} shield: {shield_info}")
        
        # steps the wrapped env
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = action
        if shield_info is not None:
            if not isinstance(info, dict):
                info = dict()
            info = dict(info)
            info['shield'] = shield_info
        
        # collect the sample for monitoring 
        s = self.get_sample(obs, action, reward)
        
        # add sample and compute robustness
        self.stl_driver.add_sample(s,False)

        # Monitor formulas: evaluated once per step, whether or not they also
        # act as obs/reward formulas (non-zero obs_weight/reward_weight).
        # Each one always produces a *label* for the reward machines: a label
        # is true when its monitored lower robustness bound is > 0. A monitor
        # formula with 'terminal: True' also terminates the episode when it
        # becomes true. The recorded value (res_f['monitor_formulas/<f>']) is
        # the evaluated formula value (per its 'semantics' option).
        mon_vals = {}
        labels = []
        for f_name, f_opt in self.monitor_formulas.items():
            val, eval_res = self.eval_formula_cfg(f_name, f_opt)
            mon_vals[f_name] = val
            self.episode['res_f'][f'monitor_formulas/{f_name}'].append(val)
            if eval_res['lower_rob'] > 0:
                labels.append(f_name)
                terminal = (f_opt or {}).get('terminal', False)
                if terminal:
                    print('Episode terminated because of formula', f_name)
                    terminated = True
            if self.debug_formulas is True:
                print(f'monitor {f_name}: {float(val):.3}', end=' ')
        if self.monitor_formulas and self.debug_formulas is True:
            print('')
        # formulas true *now* act on the agent's action from the next step
        # on (their truth is only known after the env step, mirroring how
        # reward-machine states persist)
        self._active_formula_actions = [
            f for f in self.monitor_actions if f in set(labels)
        ]

        idx_formula = 0
        num_obs_formulas= len(self._obs_entries)
        robs = [0]*num_obs_formulas
        for f_name, f_opt, prefix in self._obs_entries:                          
            if prefix == 'monitor_formulas':
                # already evaluated in the monitor pass; scale by obs_weight
                robs_f = mon_vals[f_name] * f_opt['obs_weight']
            else:
                robs_f,_ = self.eval_formula_cfg(f_name,f_opt)        
            robs[idx_formula] = robs_f # forget about low and high robs for now                        
            if prefix != 'monitor_formulas':
                # monitor entries are recorded once in the monitor pass above
                self.episode['res_f'][f'{prefix}/{f_name}'].append(robs_f)
            idx_formula+=1     
            if self.debug_formulas is True:                
                if idx_formula==1:
                    print('obs formulas', end='  --  ')                
                print(f_name+f': {float(robs_f):.3}', end=' ')
                if idx_formula==num_obs_formulas:
                    print('')
        
        # Step every reward machine with the current true-label set.
        rm_states = []
        rm_rewards = []
        label_set = set(labels)
        for rm in self.reward_machines:
            m = rm['machine']
            if self.debug_rm:
                before = m.state_id
                activated = [
                    t for t in m.outgoing.get(before, [])
                    if t['condition'] in label_set
                ]
                act = ", ".join(t['id'] for t in activated) or "(none -> stay)"
                print(f"t:{self.time_step} rm '{rm['name']}': state={before} "
                      f"labels={labels} activated=[{act}]")
            r = m.step(labels)
            rm_states.append(rm['machine'].state_idx)
            rm_rewards.append(r)
            if self.debug_rm:
                print(f"        -> state={m.state_id} reward={r}")
            if rm['machine'].is_final:
                terminated = True
        self.episode['labels'].append(labels)
        self.episode['rm_states'].append(rm_states)
        self.episode['rm_rewards'].append(rm_rewards)
        rm_states_array = np.array(rm_states, dtype=np.int_)

        if not self.multi_objective:
            # The returned new_reward is a scalar
            new_reward = reward * self.env_reward_weight
            # add stl robustness to reward
            for f_name, f_opt, prefix in self._reward_entries:                         
                if prefix == 'monitor_formulas':
                    # already evaluated in the monitor pass
                    robs_f = mon_vals[f_name]
                else:
                    robs_f,_ = self.eval_formula_cfg(f_name,f_opt)            
                if prefix != 'monitor_formulas':
                    # monitor entries are recorded once in the monitor pass
                    self.episode['res_f'][f'{prefix}/{f_name}'].append(robs_f)
                f_opt = f_opt or {}
                w = f_opt.get('reward_weight', f_opt.get('weight', 1))        
                new_reward += w*robs_f   
            # add weighted reward-machine rewards to the overall reward
            for rm, r in zip(self.reward_machines, rm_rewards):
                new_reward += rm['weight'] * r
        else:
            # The returned new_reward is a vector
            new_reward = [reward * self.env_reward_weight] if self.env_reward_weight else []
            for f_name, f_opt, prefix in self._reward_entries:                         
                if prefix == 'monitor_formulas':
                    robs_f = mon_vals[f_name] * f_opt['reward_weight']
                else:
                    robs_f,_ = self.eval_formula_cfg(f_name,f_opt)            
                if prefix != 'monitor_formulas':
                    self.episode['res_f'][f'{prefix}/{f_name}'].append(robs_f)
                new_reward.append(robs_f)
            new_reward = np.asarray(new_reward)
        
        # update current time
        self.time_step += 1
        self.current_time += self.real_time_step
        
        # return obs with added robustness / rm states
        new_obs = self._build_obs(obs, robs, rm_states_array)

        self.episode['observations'].append(self.last_obs)               
        self.episode['actions'].append(action)
        self.episode['env_rewards'].append(reward)
        self.episode['rewards'].append(new_reward)
        self.episode['dones'].append(terminated)
        self.episode['last_obs'] = new_obs
        self.last_obs = new_obs    
        
        return new_obs, new_reward, terminated, truncated, info

    def get_sample(self,obs,action,reward):        
        # get a sample for stl driver, converts obs, action,reward into (t, signals_values)
   
        s = np.zeros(len(self.signals_specs)+1)
        s[0] = self.current_time
        if self.debug_signals is True:
                print()
                print(f't:{s[0]:.3}',end=' ')
        i_sig = 0
        
        for sig in self.signals_specs:
            value= self.signals_map[sig]
            i_sig = i_sig+1           
            signal = eval(value)
            if isinstance(signal, np.ndarray):
                assert signal.size == 1, "Signal must be float or an np.ndarray with a single element"
                signal = signal.flatten()[0]
            s[i_sig] = signal
            if self.debug_signals is True:
                print(f'{sig}: {s[i_sig]:.3}', end=' ')
        if self.debug_signals is True:
            print()
        return s
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.reset(seed=seed)

    def get_ep_len(self):
        stl_data = self.episode.get('stl_data')
        # stl_data is a stlrom.STLData (live handle on the driver's data);
        # it is only set after the first reset (before that it may be missing
        # or still the initial empty list)
        if not isinstance(stl_data, stlrom.STLData):
            return 0
        return len(stl_data.get_signal(0).get_samples_list())

    def get_time(self):
        stl_data = self.episode.get('stl_data')
        if not isinstance(stl_data, stlrom.STLData):
            raise ValueError("No data to plot.")
        samples = stl_data.get_signal(0).get_samples_list()
        return [s.time for s in samples]
        
    def get_sig(self, sig_name):
        # recover a signal computed during an episode, either observation, reward, reward formula, 
        sig= stlrom.Signal()
        if self.get_ep_len()>0:
            if sig_name in self.signals_specs:
                sig= self.episode['stl_data'].get_signal(sig_name)   
        return sig

    def _formula_cfg_weight(self, prefix, fname):
        """Configured weight (default 1.0) of formula `fname` in group `prefix`.

        Only the groups that carry rewards / contribute to the wrapped reward
        expose a meaningful `weight` ('reward_formulas', 'monitor_formulas',
        'obs_formulas'); other groups default to 1.0. For monitor formulas the
        effective weight is `reward_weight` if non-zero, else `obs_weight` if
        non-zero, else the generic `weight` (default 1.0). The legacy
        'label_formulas' prefix is treated like 'monitor_formulas'.
        """
        if prefix in ('label_formulas', 'monitor_formulas'):
            grp = self.monitor_formulas
            opt = grp.get(fname)
            if isinstance(opt, dict):
                for key in ('reward_weight', 'obs_weight'):
                    w = opt.get(key, 0)
                    if w:
                        return w
                return opt.get('weight', 1.0)
            return 1.0
        if prefix == 'reward_formulas':
            grp = self.reward_formulas
        elif prefix == 'obs_formulas':
            grp = self.obs_formulas
        else:
            return 1.0
        opt = grp.get(fname)
        if isinstance(opt, dict):
            return opt.get('weight', 1.0)
        return 1.0

    def _formula_group_of(self, fname):
        """Name of the formula group owning `fname`, or None if it is not a
        monitored/recorded formula."""
        for prefix, grp in (('reward_formulas', self.reward_formulas),
                            ('monitor_formulas', self.monitor_formulas),
                            ('obs_formulas', self.obs_formulas)):
            if fname in grp:
                return prefix
        return None

    def _res_f_trace(self, fname):
        """Recorded per-step trace of a formula from res_f (namespaced keys),
        or None if it was not recorded. If a formula appears in several
        groups, 'reward_formulas' takes precedence, then monitor/obs/end.
        """
        for prefix in ('reward_formulas', 'monitor_formulas', 'obs_formulas'):
            key = f'{prefix}/{fname}'
            if key in self.episode['res_f']:
                return self.episode['res_f'][key]
        return None

    def _reward_machine_idx(self, name):
        idx = next((i for i, rm in enumerate(self.reward_machines)
                    if rm.get('name') == name), None)
        if idx is None:
            raise ValueError(
                f"Unknown reward machine '{name}'. Available reward machines: "
                f"{[rm.get('name') for rm in self.reward_machines]}")
        return idx

    def _resolve_signal_str(self, name):
        """Resolve a plot/signal string to (values, sig_type, discrete, sig).

        Returns a 1-D sequence of values, a ``sig_type`` in
        {'val', 'rob', 'sat'}, a boolean flag telling whether the signal is
        discrete (zero-order-hold / step), and a ``stlrom.Signal`` when the
        token maps directly to one (raw mapped signals) -- otherwise the
        fourth element is None and the values must be re-encoded into a
        Signal (see get_signal_from_str). See CODEBASE_OVERVIEW for the full
        grammar supported:

          * bare mapped signal names (action_names / obs_names /
            aux_sig_names / env_reward)
          * 'reward' -> the agent-visible (shaped) reward
          * '<prefix>/<f>'  prefix in monitor_formulas, obs_formulas,
            reward_formulas (legacy), label_formulas (legacy alias),
            eval_formulas
          * 'rm_states/<m>' and 'rm_reward/<m>' (or bare machine name <m>)
          * 'rho(<f>[,h])', 'rob(<f>[,h])', 'sat(<f>)'
          * bare <f> (formula) -> offline robustness

        An optional ':weighted' suffix multiplies the values by the relevant
        configured weight (formula weight, reward-machine weight, or
        `env_reward_weight` for the `env_reward` raw signal). For
        `env_reward:weighted` the returned Signal is a re-encoded, scaled
        copy (the driver's Signal itself cannot be scaled).
        """
        weighted = False
        base = name
        if base.endswith(':weighted'):
            weighted = True
            base = base[:-len(':weighted')]

        sig_type = 'val'
        discrete = False

        # -- function forms -------------------------------------------------
        for fn, ftype in (('rho(', 'rob'), ('rob(', 'rob'), ('sat(', 'sat')):
            if base.startswith(fn):
                arg_rho = base[len(fn):-1].split(',')
                past_horizon = 0
                phi = arg_rho[0]
                if len(arg_rho) > 1:
                    past_horizon = -float(arg_rho[1])
                values = self.get_rob(phi, horizon=past_horizon, online=False)
                sig_type = ftype
                discrete = (ftype == 'sat')
                if ftype == 'sat':
                    values = (values > 0).astype(int)
                return np.asarray(values, dtype=float), sig_type, discrete, None

        # -- prefix/name forms ----------------------------------------------
        if '/' in base:
            prefix, sub = base.split('/', 1)

            if prefix == 'rm_states':
                i = self._reward_machine_idx(sub)
                values = [step[i] for step in self.episode['rm_states']]
                return np.asarray(values, dtype=float), 'val', True, None

            if prefix in ('rm_reward',):
                i = self._reward_machine_idx(sub)
                values = [step[i] for step in self.episode['rm_rewards']]
                w = self.reward_machines[i].get('weight', 1.0)
                values = np.asarray(values, dtype=float)
                if weighted:
                    values = values * w
                return values, 'val', True, None

            if prefix in ('obs_formulas', 'reward_formulas', 'label_formulas',
                          'monitor_formulas'):
                # legacy plot strings: 'label_formulas/<f>' points at the
                # monitor group, 'obs/reward_formulas/<f>' fall back to a
                # monitor entry that plays that role
                if prefix == 'label_formulas':
                    key = f'monitor_formulas/{sub}'
                else:
                    key = f'{prefix}/{sub}'
                    if key not in self.episode['res_f']:
                        key = f'monitor_formulas/{sub}'
                if key not in self.episode['res_f']:
                    raise ValueError(f"Unknown formula '{sub}' in '{name}'.")
                values = np.asarray(self.episode['res_f'][key], dtype=float)
                if weighted:
                    w = self._formula_cfg_weight(prefix, sub)
                    values = values * w
                # monitor formulas hold per-step evaluated values (0/1-ish for
                # Boolean semantics); treat as discrete-steps like states.
                return values, 'rob', (prefix in ('label_formulas', 'monitor_formulas')), None

            if prefix == 'eval_formulas':
                values = self.get_rob(sub, online=False)
                return np.asarray(values, dtype=float), 'rob', False, None

        # -- bare names -----------------------------------------------------
        if base == 'reward':
            # agent-visible (shaped) reward, the actual SpecWrapper output.
            values = np.asarray(self.episode['rewards'], dtype=float)
            return values, 'val', True, None

        if base == 'env_reward':
            sig = self.get_sig('env_reward')
            values = np.asarray([s.value for s in sig.get_samples_list()], dtype=float)
            if weighted:
                # env_reward:weighted -> raw reward scaled by env_reward_weight;
                # the scaled trace is re-encoded (Signal has no scalar mul, and
                # returning the driver's Signal as-is would lose the scaling)
                return values * self.env_reward_weight, 'val', True, None
            return values, 'val', True, sig

        # bare reward-machine name -> its per-step (unweighted) reward
        for rm in self.reward_machines:
            if rm.get('name') == base:
                i = self._reward_machine_idx(base)
                values = [step[i] for step in self.episode['rm_rewards']]
                values = np.asarray(values, dtype=float)
                if weighted:
                    values = values * rm.get('weight', 1.0)
                return values, 'val', True, None

        # mapped raw signals (obs/action/aux...): the token maps directly to
        # the driver's stlrom.Signal, which is returned as-is (its own
        # plotting handles interpolation); values are extracted from its
        # samples for value-based consumers (live plots, ...). ('env_reward'
        # is handled above, including ':weighted'.) ':weighted' has no effect
        # on the other raw signals.
        env_signal_names = self.signals_map.keys()
        if base in env_signal_names:
            sig = self.get_sig(base)
            values = np.asarray([s.value for s in sig.get_samples_list()], dtype=float)
            discrete = (base == 'env_reward'
                        or base in getattr(self, '_action_signal_names', set()))
            return values, 'val', discrete, sig

        # bare formula name: prefer the per-step values already recorded in
        # res_f (reward/label/obs/end formulas), otherwise compute offline
        # robustness over the whole trace.
        trace = self._res_f_trace(base)
        if trace is not None:
            values = np.asarray(trace, dtype=float)
            if weighted:
                group = self._formula_group_of(base)
                w = self._formula_cfg_weight(group, base) if group else 1.0
                values = values * w
            return values, 'rob', False, None
        try:
            values = self.get_rob(base, online=False)
            return np.asarray(values, dtype=float), 'rob', False, None
        except Exception:
            pass

        # unknown -> error so get_fig reports a clear warning
        raise ValueError(
            f"Cannot resolve signal/formula name '{name}'. It is neither a "
            "declared signal, a formula, a reward machine nor a known "
            "plot token (see get_signal_from_str docs).")

    def get_values_from_str(self, str):
        # Backward-compatible value lookup; delegates to the shared resolver.
        values, sig_type, _discrete, _sig = self._resolve_signal_str(str)
        return values, sig_type

    def get_signal_from_str(self, str):
        """Return an ``stlrom.Signal`` for a plot/signal name string.

        Raw mapped signals (obs/actions/aux, ``env_reward``) are returned
        directly as the driver's ``stlrom.Signal`` -- its own plotting
        handles interpolation. Other tokens (rewards, formula robustness,
        RM states/rewards, ...) are re-encoded into a fresh ``Signal``:
        discrete (zero-order-hold) signals -- rewards, actions, label/states,
        sat(...) -- are encoded with constant samples (derivative 0) so they
        render as steps; continuous signals use linear interpolation.
        """
        values, _sig_type, discrete, sig = self._resolve_signal_str(str)
        if sig is not None:
            return sig
        times = self.get_time()
        values = np.asarray(values, dtype=float)
        if len(times) != len(values):
            # pad/trim robustness vectors that lag the trace length
            n = min(len(times), len(values))
            times, values = times[:n], values[:n]
        S = stlrom.Signal()
        for t, v in zip(times, values):
            if discrete:
                S.append_constant_sample(float(t), float(v))
            else:
                S.append_linear_sample(float(t), float(v))
        if times:
            S.end_time = float(times[-1])
        return S

    def get_rob(self, formula, horizon=0, online=True):
    # Compute robustness signal. If online is true, then 
    # compute it at each time as if future was not known, 
    # otherwise uses all data for all computation

        data = self.stl_driver.data
        times = [s.time for s in data.get_signal(0).get_samples_list()]
        if len(times) == 0:
            raise ValueError("No data/episode was computed.")

        # stlrom does not raise for unknown formulas (it prints a warning and
        # returns 0. / an empty list), so validate up front for a clear error
        if self.stl_driver.get_monitor(formula).formula is None:
            raise ValueError(f"Unknown formula '{formula}'.")

        rob = np.zeros(len(times))
        if online:
            # Replay the trace sample-by-sample into a fresh driver so each
            # step's online robustness is computed without the future
            # (stlrom has no reset API, and the driver's data can only be
            # appended to).
            replay = stlrom.STLDriver()
            replay.parse_string(self.stl_specs_str)
            n_sig = len(self.signals_specs)
            sig_samples = [data.get_signal(i).get_samples_list() for i in range(n_sig)]
            for k, t in enumerate(times):
                row = [t] + [sig_samples[i][k].value for i in range(n_sig)]
                replay.add_sample(row, False)
                t0 = max(0., t - horizon)
                robs = replay.get_online_rob(formula, t0)
                rob[k] = robs[0] if len(robs) > 0 else 0.
        else:
            # offline: the whole trace is known; evaluate the formula at
            # each time of the trace
            for k, t in enumerate(times):
                rob[k] = self.stl_driver.get_rob(formula, max(0., t - horizon))
        return rob

    def set_episode_data(self, episode):
        self.reset_monitor()
        self.episode = episode
        self.stl_driver.data = self.episode['stl_data']
        self._active_formula_actions = []
        observations = self.episode['observations']
        
        self.time_step = self.get_ep_len()-1
        self.current_time = self.time_step*self.real_time_step
           
    def eval_formula_cfg(self, f_name, f_opt, eval_res=None):
    # eval a formula based on f_opt configuration options AT CURRENT STEP
    # it uses whatever is in the data of stl_driver at this point

        if f_opt is None:
            f_opt={}        
        # 'formula' (preferred) / 'formula_name' (legacy) select the actual STL
        # formula to evaluate; when absent, the config key is the formula name.
        # This lets the same formula be evaluated under several configurations
        # (different past_horizon, t0, bounds, formula_params, ...).
        f_name = f_opt.get('formula', f_opt.get('formula_name', f_name))
        f_hor = f_opt.get('past_horizon',0.)   
        t0_cfg = f_opt.get('t0')
        tend = f_opt.get('tend',self.current_time)
        # Anchoring of the evaluation time:
        #  - default (no 'online'/'eval_all_steps'/'t0' option): at the
        #    current time (minus past_horizon), i.e. the formula's value
        #    "now". This keeps the per-step records (res_f, reward shaping,
        #    labels, obs formulas) consistent with rho(f)/rob(f) plots;
        #  - 'online: True' or 'eval_all_steps: True': explicit same anchoring;
        #  - explicit 't0' (without 'online') or 'online: False': fixed t0
        #    (default 0) -- e.g. whole-episode formulas tracked from the start.
        online = f_opt.get('online', f_opt.get('eval_all_steps'))
        if online is None:
            online = t0_cfg is None
        if online:
            t0 = max(t0_cfg or 0., tend-f_hor)
        else:
            t0 = t0_cfg or 0.
        # FIXME: at the start, t0 cannot be negative: if past_horizon is more
        # than current_time, max() falls back to the configured t0 (0 by
        # default) and the interpretation of past formulas may differ from
        # the intended one. Eventually we should either force past formulas,
        # or provide another option: 'offline', 'online' (as now), 'past'
        # Apply per-config formula parameters (e.g. close_dist: 3.0). stlrom
        # parameters are driver-global, so save and restore them around the
        # evaluation; different configs of the same formula can thus use
        # different parameter values. Params must be declared in the spec
        # string ('param <name> = <value>').
        f_params = f_opt.get('formula_params') or {}
        saved_params = [(p, self.stl_driver.get_param(p)) for p in f_params]
        for p, v in f_params.items():
            self.stl_driver.set_param(p, float(v))
        try:
            robs = self.stl_driver.get_online_rob(f_name, t0)
        finally:
            for p, v in saved_params:
                self.stl_driver.set_param(p, v)
        val = robs[0]
        if eval_res is None:
            eval_res= dict()
            eval_res['estimate_rob'] = np.array(robs[0])
            eval_res['lower_rob'] = np.array(robs[1])
            eval_res['upper_rob'] = np.array(robs[2])                        
            sat = 1 if robs[0]>0 else 0
            eval_res['sat'] = np.array(sat)
        else:            
            eval_res['estimate_rob'] = np.append(eval_res['estimate_rob'],robs[0])
            eval_res['lower_rob'] = np.append(eval_res['lower_rob'],robs[1])
            eval_res['upper_rob'] = np.append(eval_res['upper_rob'],robs[2])                        
            sat = 1 if robs[0]>0 else 0
            eval_res['sat'] = np.append(eval_res['sat'],sat)
        
        semantics = f_opt.get('semantics', 'rob')
        if semantics == 'lower_rob':
            val = robs[1]
        elif semantics == 'upper_rob':
            val = robs[2]
        elif semantics == 'bool':
            if robs[1]>0:
                val=1
            else:
                val=0
        
        upper_bound = f_opt.get('upper_bound',np.inf)
        lower_bound = f_opt.get('lower_bound',-np.inf)
        val = max(val, lower_bound)
        val = min(val, upper_bound)
        
        if self.debug_formulas>=2:
            print(f'{f_name} at t0={t0} (current_time:{tend}, f_hor:{f_hor}): [ {robs[1]}  <=  {val} <= {robs[2]}]')
        
        return val, eval_res
               
    def eval_specs_episode(self, episode=None, res=dict(), res_rew_f_list=[], res_eval_f_list=[]):
    # computes different metrics to evaluate an episode. If res contains values already, concatenate
    # returns top level metrics, and robustness and stuff at all steps for reward formulas and eval formulas    
        
        if episode is None:
            episode= self.episode            
        else:
            self.episode = episode
                
        rewards = episode['rewards']        

        # episode length
        ep_len = len(rewards)        
        res = add_metric(res,'ep_len',ep_len)
        
        # cumulative reward
        ep_rew=0        
        for step in range(0,ep_len):            
            ep_rew +=  rewards[step]
        res= add_metric(res,'ep_rew',ep_rew)

        res_all_ep = dict({'basics':{}, 'reward_formulas':dict(), 'eval_formulas':dict()})            
        res_all_ep['basics']['mean_ep_len'] = np.double(res['ep_len']).mean()
        res_all_ep['basics']['mean_ep_rew'] = np.double(res['ep_rew']).mean()
        # fraction of episodes that ran the full num_steps (not terminated early)
        num_steps = int((self.cfg.get('cfg_test') or {}).get('num_steps', 100))
        res_all_ep['basics']['completed'] = (np.double(res['ep_len']) == num_steps).mean()
        # maybe a mean mean reward ?
        
        stl_data = episode['stl_data']            
        len_episode = len(stl_data.get_signal(0).get_samples_list())
        if self._reward_entries:
            
            res_f = episode['res_f']
            # Summarize
            for f_name, f_cfg, prefix in self._reward_entries:
                if f_name not in res:
                    res[f_name] = dict()
                f_cfg = f_cfg or {}
                w = f_cfg.get('reward_weight', f_cfg.get('weight', 1))
                res[f_name] = add_metric(res[f_name], 'mean', w*np.double(res_f[f'{prefix}/{f_name}']).mean())
                
                sum_f = w*np.double(res_f[f'{prefix}/{f_name}']).sum()
                res[f_name] = add_metric(res[f_name], 'sum', sum_f)
                
                num_sat = (np.double(res_f[f'{prefix}/{f_name}'])>0).sum()
                res[f_name] = add_metric(res[f_name], 'num_sat', num_sat)

            res_rew_f_list.append(res_f)            

        # eval formulas - for those, we evaluate off-line, after the trace has been completely computed            
        if self.eval_formulas != dict():
                        
            self.current_time=0
            self.time_step = 0            

            res_f = dict()
            for f_name,f_cfg in self.eval_formulas.items():
                res_f[f_name] = []
            
            # eval formulas at all steps 
            while self.time_step<len_episode:
                
                # eval all formulas for this step - possible optimization here is to do only first step when eval_all_steps is false
                for f_name,f_cfg in self.eval_formulas.items():
                    # if formula is new, create field in res for global evaluation
                    if f_name not in res:                        
                        res[f_name] = dict() 
                    
                    # compute and append formula eval for this step 
                    v, _ = self.eval_formula_cfg(f_name,f_cfg)    
                    res_f[f_name] = np.append(res_f[f_name],v)                                        
                    
                self.time_step +=1
                self.current_time += self.real_time_step
            
            # Synthesize: we compute all metrics - maybe we choose in cfg  (TODO)
            for f_name,f_cfg in self.eval_formulas.items():                
                if f_cfg is None:
                    f_cfg = {}
                eval_all_steps = f_cfg.get('eval_all_steps', False) or f_cfg.get('online', False)                
                if eval_all_steps:
                    w = f_cfg.get('weight', 1)
                    res[f_name] = add_metric(res[f_name], 'mean', w*res_f[f_name].mean())               
                    sum_f = w*res_f[f_name].sum()
                    res[f_name] = add_metric(res[f_name], 'sum', sum_f)
                    num_sat = (res_f[f_name]>0).sum()
                    res[f_name] = add_metric(res[f_name], 'num_sat', num_sat)                
                else:
                    init_rob = res_f[f_name][0]
                    res[f_name] = add_metric(res[f_name], 'init_rob', init_rob)
                    init_sat = 1 if res_f[f_name][0]>0 else 0
                    res[f_name] = add_metric(res[f_name], 'init_sat', init_sat)                
                
            res_eval_f_list.append(res_f)
                        
            for f_name, _f_cfg, _prefix in self._reward_entries:
              if isinstance(res[f_name], dict):                    
                res_all_ep['reward_formulas'][f_name] = dict()
                res_all_ep['reward_formulas'][f_name]['mean_rob'] = res[f_name]['mean'].mean()
                res_all_ep['reward_formulas'][f_name]['mean_num_sat'] = res[f_name]['num_sat'].mean()
                res_all_ep['reward_formulas'][f_name]['mean_sum'] = res[f_name]['sum'].mean()

            num_ep = len(res['ep_len'])
            for f_name,f_cfg in self.eval_formulas.items():
              if f_cfg is None:
                    f_cfg = {}                
              if isinstance(res[f_name], dict):                    
                res_all_ep['eval_formulas'][f_name] = dict()                
                eval_all_steps = f_cfg.get('eval_all_steps', False) or f_cfg.get('online', False)           
                if eval_all_steps:
                    res_all_ep['eval_formulas'][f_name]['mean_sum'] = res[f_name]['sum'].mean()
                    res_all_ep['eval_formulas'][f_name]['mean_mean'] = res[f_name]['mean'].mean()
                    res_all_ep['eval_formulas'][f_name]['mean_num_sat'] = res[f_name]['num_sat'].mean()
                else:
                    res_all_ep['eval_formulas'][f_name]['ratio_init_sat'] = (res[f_name]['init_sat']>0).sum()/num_ep
                    res_all_ep['eval_formulas'][f_name]['mean_init_rob'] = res[f_name]['init_rob'].mean()

                        

        return res, res_all_ep, res_rew_f_list, res_eval_f_list
    

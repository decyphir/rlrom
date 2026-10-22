import gymnasium as gym
import numpy as np


def _trans_id(from_id, to_id, condition):
    return f"{from_id}__to__{to_id}__if__{condition}"


class RewardMachine:
    """A pure finite-state machine (FSM) defining reward-machine dynamics.

    This class is intentionally independent of gym wrappers and STL evaluation:
    it only parses a ``cfg_rm`` dict, tracks the current state, consumes a list of
    *true* condition labels at each step, and decides a transition based on priority.
    It knows nothing about how those labels are computed (STL robustness, plain
    booleans, ...) -- the caller (wrapper) is responsible for building the label list.

    Semantics of ``step(labels)``:
      * let ``A`` be the outgoing transitions from the current state whose
        ``condition`` is in ``labels`` (i.e. activated);
      * if ``A`` is empty, stay in the current state and issue that state's reward;
      * otherwise, take the activated transition with the highest ``priority``
        (default 0) and issue that transition's ``reward`` (default 0).
      * if two or more activated transitions share the same (max) priority,
        a ``ValueError`` is raised (non-determinism guard).

    Config contract (compatible with existing examples, e.g. minigrid):
      states:      list of dicts: ``id`` (str), optional ``initial`` (bool),
                   optional ``final`` (bool), optional ``reward`` (default 0).
                   Exactly one state must be ``initial``.
      transitions: list of dicts: ``from``, ``to``, ``condition`` (all single
                   strings), optional ``reward`` (default 0), optional ``priority``
                   (default 0), optional ``id`` (default ``{from}__to__{to}__if__{condition}``).

    A state that is ``final`` marks the end of the episode once reached. Note that
    ``final`` and ``initial`` may coexist on the same state.
    """

    def __init__(self, cfg_rm):
        self.cfg_rm = cfg_rm
        states = cfg_rm.get('states', [])
        if not states:
            raise ValueError("RewardMachine: cfg_rm must define at least one state")

        # Parse states.
        self.states = []
        self.states_idx_map = {}   # state id -> index; index -> state id
        self._is_initial = {}
        self._is_final = {}
        self.state_rewards = {}
        initial_states = []
        for i, s in enumerate(states):
            sid = s['id']
            if sid in self.states_idx_map:
                raise ValueError(f"RewardMachine: duplicate state id '{sid}'")
            self.states.append(sid)
            self.states_idx_map[sid] = i
            self.states_idx_map[i] = sid
            self._is_initial[sid] = bool(s.get('initial', False))
            self._is_final[sid] = bool(s.get('final', False))
            self.state_rewards[sid] = s.get('reward', 0)
            if self._is_initial[sid]:
                initial_states.append(sid)

        if len(initial_states) != 1:
            raise ValueError(
                f"RewardMachine: expected exactly one initial state, got {initial_states}"
            )
        self.u_0 = initial_states[0]

        # Parse transitions.
        self.transitions = []
        self.outgoing = {}   # from state id -> list of parsed transition dicts
        for t in cfg_rm.get('transitions', []):
            for field in ('from', 'to', 'condition'):
                if field not in t:
                    raise ValueError(f"RewardMachine: transition missing required field '{field}': {t}")
            entry = {
                'from': t['from'],
                'to': t['to'],
                'condition': t['condition'],
                'reward': t.get('reward', 0),
                'priority': t.get('priority', 0),
                'id': t.get('id', _trans_id(t['from'], t['to'], t['condition'])),
            }
            if entry['from'] not in self.states_idx_map:
                raise ValueError(f"RewardMachine: transition from unknown state '{entry['from']}'")
            if entry['to'] not in self.states_idx_map:
                raise ValueError(f"RewardMachine: transition to unknown state '{entry['to']}'")
            self.transitions.append(entry)
            if entry['from'] not in self.outgoing:
                self.outgoing[entry['from']] = []
            self.outgoing[entry['from']].append(entry)

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Return to the unique initial state."""
        self.u_in = self.u_0
        return self.u_in

    @property
    def state_id(self):
        return self.u_in

    @property
    def state_idx(self):
        return self.states_idx_map[self.u_in]

    @property
    def is_final(self):
        """True once the machine is in a state marked ``final``."""
        return self.is_final_state(self.u_in)

    def is_final_state(self, state_id):
        return self._is_final.get(state_id, False)

    # ------------------------------------------------------------------ step
    def step(self, labels):
        """Consume a list of true condition labels; take a transition if any.

        Returns the reward issued this step: the transition reward if an activated
        transition is taken, otherwise the current state's reward (default 0).
        The current state is updated as a side effect.
        """
        if labels is None:
            labels = []
        labels = set(labels)

        activated = [
            t for t in self.outgoing.get(self.u_in, [])
            if t['condition'] in labels
        ]

        if not activated:
            # stay in place; issue the state reward.
            return self.state_rewards[self.u_in]

        # pick the highest priority; error on a tie among the selected ones.
        max_priority = max(t['priority'] for t in activated)
        best = [t for t in activated if t['priority'] == max_priority]
        if len(best) > 1:
            raise ValueError(
                "RewardMachine: ambiguous transition -- several activated transitions "
                f"share max priority {max_priority} from state '{self.u_in}': "
                + ", ".join(t['id'] for t in best)
            )
        chosen = best[0]
        self.u_in = chosen['to']
        return chosen['reward']



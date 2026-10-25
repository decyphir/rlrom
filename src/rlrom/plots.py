import re
import matplotlib
import matplotlib.pyplot as plt
import rlrom.utils as utils

def plot_signal(env, st, ax):
    get_sig = env.get_wrapper_attr('get_signal')
    sig=  get_sig(st)
    sig.plot(ax=ax)
    ax.grid()

def plot_rob(env, formula, ax, past_horizon=0, online=True):
    get_time = env.get_wrapper_attr('get_time')
    get_rob = env.get_wrapper_attr('get_rob')
    time = get_time()
    vals = get_rob(formula, horizon=past_horizon, online= online)
    ax.plot(time, vals, label = formula)
    ax.legend()
    ax.grid()

def _is_discrete_signal(env, st):
    """True if the stlrom Signal backing `st` has derivative 0 everywhere
    (i.e. a zero-order-hold / step signal such as actions, labels, states)."""
    try:
        S = env.get_wrapper_attr('get_signal_from_str')(st)
    except Exception:
        return False
    samples = S.get_samples_list()
    if not samples:
        return False
    return all(sam.derivative == 0 for sam in samples)


class live_line:
    def __init__(self, T, st, ax):
        self.env = T.env
        self.get_time = T.env.get_wrapper_attr('get_time')
        get_vals_from_st = T.env.get_wrapper_attr('get_values_from_str')
        self.get_vals = lambda: get_vals_from_st(st)
        self.ax = ax
        self.label = st
        self.line = self.reset_plot()

    def reset_plot(self):
        kwargs = {'label': self.label}
        # Discrete signals are drawn as steps, matching Signal.plot.
        if _is_discrete_signal(self.env, self.label):
            kwargs['drawstyle'] = 'steps-post'
        return self.ax.plot([], [], **kwargs)[0]

    def update(self):
        t = self.get_time()
        v,_ = self.get_vals()
        self.line.set_xdata(t)
        self.line.set_ydata(v)
        self.line.set_label(self.label)
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        self.ax.legend()
  
class RLFig: 
    def __init__(self, T, layout):
        plt.ioff()
        self.tester = T
        self.layout = get_layout_from_string(layout)        
        N = len(self.layout)
        fig, axs = plt.subplots(N, 1, sharex=True, figsize=(9, 2*N))
                
        self.live_lines = []
        iax = 0
        for l in self.layout:
            for s in l:
                self.live_lines.append(live_line(T, s, axs[iax]))
            axs[iax].grid()
            iax +=1
        self.fig = fig
        self.axs= axs
        self.tester.callbacks.append(self.update)

    def update(self):
        l0 = self.live_lines[0]
        t = l0.get_time()
        if len(t) == 0:
            return  # no samples yet (first step)
        self.axs[0].set_xlim(max(0, len(t)-25), len(t))
        for l in self.live_lines:
            l.update()
        self.fig.canvas.draw()

def get_layout_from_string(signals_layout):
    out = []
    # split signals string wrt linebreaks first
    signals_rows = signals_layout.splitlines()

    # then strip and split wrt commas
    # A signal name is an identifier, optionally with a 'prefix/name' part and a
    # ':option' suffix (e.g. 'label_formulas/go_left', 'rm_reward/left_right:weighted'),
    # optionally followed by a '(...)' argument list (e.g. 'rho(phi, 50)').
    sig_re = r'[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)?(?::[A-Za-z_][A-Za-z0-9_]*)?(?:\([^)]*\))?'
    for line in signals_rows:        
        if line.strip() == '':
            continue
        else:
            out_row = re.findall(sig_re, line)            
            out.append(out_row)        

    return out            

def plot_envelope(steps, mean_val, min_val, max_val, 
                      ax=None, label=None, color='blue', linestyle='-'):
    linestyle_mean = '-'

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
        ax.grid(True)
    if label is None:    
        ax.plot(steps, mean_val, color=color,linestyle=linestyle_mean)
    else:
        ax.plot(steps, mean_val,label=label, color=color,linestyle=linestyle_mean)

    ax.plot(steps, min_val, color=color, linestyle=linestyle)
    ax.plot(steps, max_val, color=color,linestyle=linestyle)

    ax.fill_between(steps, min_val, max_val, color=color, alpha=0.25, label='Min-Max Range')
    ax.legend()
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Value')


    return ax

def plot_df_envelope(df, feature, ax=None, label=None, color='gray', linestyle='-'):
# assumes df is a a dataframe df of (steps,feature) concat vertically with label
    df_envelope = utils.get_df_mean_min_max_val(df, feature)
    steps = df_envelope['steps'].to_numpy()
        
    feature_mean = f'{feature}_mean'
    feature_min = f'{feature}_min'
    feature_max = f'{feature}_max'

    mean_val = df_envelope[feature_mean].to_numpy()
    min_val = df_envelope[feature_min].to_numpy()
    max_val = df_envelope[feature_max].to_numpy()

    return plot_envelope(steps, mean_val, min_val, max_val, 
                      ax=ax, label=label, color=color, linestyle=linestyle)

def plot_tb_training_logs(data, ax=None, label=None, color='gray', linestyle='-'):
# assumes data is a list of sync logs with steps and values fields  PROBABLY DEPRECATED
    
    steps = data[0].get('steps')
    mean_val = utils.get_mean_values(data)
    min_val = utils.get_lower_values(data)
    max_val = utils.get_upper_values(data)    

    return plot_envelope(steps, mean_val, min_val, max_val, 
                      ax=ax, label=label, color=color, linestyle=linestyle)


def plot_df_training(df, formula=None, metric='mean_ep_rew', 
                     label='auto', ax=None, **kargs):
    """Plot a single metric against training steps.

    df: results DataFrame (see rlrom.utils.get_training_res).
    formula and metric: either a flat column name (metric only, e.g.
    'mean_ep_rew') or a formula name plus sub-metric looked up in the
    eval/reward formulas (e.g. formula='phi_left_goal',
    metric='ratio_init_sat').
    """
    metric_spec = f'{formula}/{metric}' if formula is not None else metric
    steps = df['steps'].to_numpy()
    metric_values = utils.get_metric_series(df, metric_spec).to_numpy()

    if ax is None:
       _, ax = plt.subplots(figsize=(8, 4))
       ax.grid(True)

    if label is None:    
        ax.plot(steps, metric_values,**kargs)
    elif label=='auto':
        ax.plot(steps, metric_values,label=metric_spec,**kargs)
    else:
        ax.plot(steps, metric_values,label=label,**kargs)
    
    ax.legend()
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Value')

    return  ax


def plot_training_results(df, metrics=None, style='line', label_col='training_idx',
                          title=None, log_y=False, figsize=(9, 4), **kargs):
    """Plot training results collected in a DataFrame.

    df: results DataFrame with a 'steps' column, as returned by
    rlrom.utils.get_training_res() (one training) or
    rlrom.utils.get_df_all_trainings() (several, labeled by 'training_idx').
    metrics: a metric name or list of names (see rlrom.utils.get_metric_series).
    style: 'line' (one line per training), 'envelope' (mean/min-max band
    across trainings at each step) or 'both'.

    Returns the matplotlib Figure.
    """
    if metrics is None:
        metrics = ['mean_ep_rew']
    elif isinstance(metrics, str):
        metrics = [metrics]

    df = df.copy()
    if label_col not in df.columns:
        df[label_col] = 'training'

    n_metrics = len(metrics)
    fig, axs = plt.subplots(n_metrics, 1, sharex=True,
                            figsize=(figsize[0], figsize[1]*n_metrics), squeeze=False)
    axs = axs[:, 0]

    for iax, metric in enumerate(metrics):
        ax = axs[iax]
        ax.grid(True)
        dfm = df[['steps', label_col]].copy()
        dfm[metric] = utils.get_metric_series(df, metric)
        dfm = dfm.sort_values('steps')

        if style in ('envelope', 'both'):
            plot_df_envelope(dfm, metric, ax=ax, label='mean', linestyle='--')
        if style in ('line', 'both') or df[label_col].nunique() <= 1:
            for lab, grp in dfm.groupby(label_col, sort=False):
                ax.plot(grp['steps'], grp[metric], label=str(lab), **kargs)

        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(metric)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    return fig


class LivePlot:
    """Live-updating matplotlib figure(s) for one or more cfg_test.plot items.

    Call step() once per environment step. By default the x axis spans
    0..end of episode; with ``live_window`` (time units) it follows a
    trailing window instead.
    Keys (interactive backends only): space pauses/resumes the updates,
    Escape closes the figures (skip the episode). Under non-interactive
    backends (e.g. Agg) step() is a fast no-op.
    """

    _PAUSE = 0.01  # per-step event pump (also paces the live animation)

    def __init__(self, T, items, live_window=None):
        self.T = T
        self.items = items
        self.live_window = live_window
        self.paused = False
        self.quit = False
        self.figs = []
        self.lines_by_fig = []
        self._interactive = matplotlib.get_backend().lower() not in (
            'agg', 'module://matplotlib.backends.backend_agg')

    def create(self):
        for p in self.items:
            layout_lines = []
            for ax in p.get('ax', []):
                if isinstance(ax, (list, tuple)):
                    layout_lines.append(', '.join(str(sig) for sig in ax))
                else:
                    layout_lines.append(str(ax))
            # same tokenizer as the static path: splits 'a, b' rows, handles
            # 'prefix/name', ':option' and '(...)' argument syntax
            layout = [row for row in get_layout_from_string("\n".join(layout_lines)) if row]
            n_ax = max(len(layout), 1)
            fig, axs = plt.subplots(n_ax, 1, sharex=True,
                                    figsize=(9, 2.5*n_ax), squeeze=False)
            axs = [axs[i, 0] for i in range(n_ax)]
            if p.get('title'):
                fig.suptitle(p['title'])
            fig_lines = []
            for iax, signals in enumerate(layout):
                for sig in signals:
                    fig_lines.append(live_line(self.T, sig, axs[iax]))
                axs[iax].grid(True)
            self.figs.append(fig)
            self.lines_by_fig.append(fig_lines)
            if self._interactive:
                fig.canvas.mpl_connect('key_press_event', self._on_key)
                plt.pause(self._PAUSE)  # register figure with the interactive manager

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            print(f"live plot: {'paused' if self.paused else 'resumed'}")
        elif event.key == 'escape':
            self.quit = True
            print("live plot: skipping (escape)")

    def step(self):
        if self.quit or not self._interactive:
            return
        if not self.paused:
            get_time = self.T.env.get_wrapper_attr('get_time')
            for fig, fig_lines in zip(self.figs, self.lines_by_fig):
                t = get_time()
                if len(t) == 0:
                    continue
                if self.live_window is not None:
                    fig.axes[0].set_xlim(max(0.0, t[-1] - self.live_window), t[-1])
                else:
                    # x always spans 0..end of episode (explicit: with empty
                    # lines the first draw autoscales to matplotlib's
                    # (-0.05, 0.05) sentinel and scalex=False never fixes it)
                    fig.axes[0].set_xlim(0.0, t[-1])
                for l in fig_lines:
                    l.update()
                fig.canvas.draw()
        # pump the event loop even while paused so keys are still read
        plt.pause(self._PAUSE)

    def close(self):
        for fig in self.figs:
            try:
                plt.close(fig)
            except Exception:
                pass
        self.figs = []
        self.lines_by_fig = []

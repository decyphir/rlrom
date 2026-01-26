import re
import matplotlib.pyplot as plt
import rlrom.utils as utils

class live_line:
    def __init__(self, T, st, ax):
        self.get_time = T.env.get_wrapper_attr('get_time')
        get_vals_from_st = T.env.get_wrapper_attr('get_values_from_str')
        self.get_vals = lambda: get_vals_from_st(st)
        self.ax = ax                
        self.label = st
        self.line = self.reset_plot()
        
    def reset_plot(self):
        self.ax.plot([], [], label = self.label)
        return self.ax.lines[-1]
    
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
        self.axs[0].set_xlim(max(0, len(t)-25), len(t))
        for l in self.live_lines:
            l.update()
        self.fig.canvas.draw()

def get_layout_from_string(signals_layout):
    out = []
    # split signals string wrt linebreaks first
    signals_rows = signals_layout.splitlines()

    # then strip and split wrt commas
    for line in signals_rows:        
        if line.strip() == '':
            continue
        else:
            out_row = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b(?:\([^)]*\))?', line)            
            out.append(out_row)        

    return out            

def plot_enveloppe(steps, mean_val, min_val, max_val, 
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

def plot_df_enveloppe(df, feature, ax=None, label=None, color='gray', linestyle='-'):
# assumes df is a a dataframe df of (steps,feature) concat vertically with label
    df_enveloppe = utils.get_df_mean_min_max_val(df, feature)
    steps = df_enveloppe['steps'].to_numpy()
        
    feature_mean = f'{feature}_mean'
    feature_min = f'{feature}_min'
    feature_max = f'{feature}_max'

    mean_val = df_enveloppe[feature_mean].to_numpy()
    min_val = df_enveloppe[feature_min].to_numpy()
    max_val = df_enveloppe[feature_max].to_numpy()

    return plot_enveloppe(steps, mean_val, min_val, max_val, 
                      ax=ax, label=label, color=color, linestyle=linestyle)

def plot_tb_training_logs(data, ax=None, label=None, color='gray', linestyle='-'):
# assumes data is a list of sync logs with steps and values fields  PROBABLY DEPRECATED
    
    steps = data[0].get('steps')
    mean_val = utils.get_mean_values(data)
    min_val = utils.get_lower_values(data)
    max_val = utils.get_upper_values(data)    

    return plot_enveloppe(steps, mean_val, min_val, max_val, 
                      ax=ax, label=label, color=color, linestyle=linestyle)


def plot_df_training(df, formula=None, metric='mean_ep_rew', 
                     label='auto',ax=None,**kargs):
    
    steps = df.collect()['steps'].to_numpy()
    
    if formula is None:
        metric_values = df.collect()[metric].to_numpy()
    else:
        df_phi = df.select('label', #training id 
                           'steps',
                            formula).unnest(formula)        
        metric_values = df_phi.collect()[metric].to_numpy()



    if ax is None:
       _, ax = plt.subplots(figsize=(8, 4))
       ax.grid(True)

    if label is None:    
        ax.plot(steps, metric_values,**kargs)
    elif label=='auto':
        ax.plot(steps, metric_values,label=metric,**kargs)
    else:
        ax.plot(steps, metric_values,label=label,**kargs)
    
    ax.legend()
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Value')

    return  ax
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.palettes import Dark2_5 as palette
import itertools
import re
import matplotlib.pyplot as plt
import rlrom.utils as utils


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


def get_fig(envs, signals_layout, tr_idx=0):
# plots stuff in bokeh from a list of environments
         
    if not isinstance(envs, list):
         envs= [envs]
    current_env= envs[tr_idx]                                                 
    lay = get_layout_from_string(signals_layout)
    status = "Plot ok. Hit reset on top right if not visible."            

    #f= figure(height=200)
    figs = []
    colors = itertools.cycle(palette)    
    
    for signal_list in enumerate(lay):
        f=None
        for signal in signal_list[1]:                
            #try: 
                color=colors.__next__()                                        
                #print(signal.strip())
                if signal.strip().startswith("set_trace_idx(") or signal.strip().startswith("_tr("):            
                    tr_idx = int(signal.split('(')[1][:-1])
                    current_env= envs[tr_idx]                                                 
                else: 
                    if f is None:
                        if figs == []:
                            f = figure(height=200)
                        else:
                            f = figure(height=200, x_range=figs[0][0].x_range)
                        figs.append([f])
                    
                    ttime = current_env.get_time()                        
                    labl = signal                                                
                    if len(envs)>1:                                                            
                        labl += ', trace_idx='+str(tr_idx)

                    sig_values, sig_type = current_env.get_values_from_str(signal)
                    if sig_type == 'val':
                        f.scatter(ttime, sig_values, legend_label=labl, color=color)
                        f.line(ttime, sig_values, legend_label=labl, color=color)
                    elif sig_type == 'rob':
                        f.step(ttime, sig_values, legend_label=labl, color=color)                        
                        
                    elif sig_type == 'sat':
                        f.step(ttime, sig_values, legend_label=labl, color=color)                        
                                            
            #except:
            #     status = "Warning: error getting values for " + signal
    fig = gridplot(figs, sizing_mode='stretch_width')        
        
    return fig, status

def plot_training_logs(data, ax=None, label=None, color='gray', linestyle='-'):
# assumes data is a list of sync logs with steps and values fields    
    
    steps = data[0].get('steps')
    mean_values = utils.get_mean_values(data)
    min_values = utils.get_lower_values(data)
    max_values = utils.get_upper_values(data)    

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.grid(True)
    if label is None:    
        ax.plot(steps, mean_values, color=color,linestyle=linestyle)
    else:
        ax.plot(steps, mean_values,label=label, color=color,linestyle=linestyle)

    ax.plot(steps, min_values, color=color, linestyle=linestyle)
    ax.plot(steps, max_values, color=color,linestyle=linestyle)

    ax.fill_between(steps, min_values, max_values, color=color, alpha=0.25, label='Min-Max Range')
    ax.legend()
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Average cumulated reward')


    return ax
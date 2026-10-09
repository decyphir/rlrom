from context import *
from pprint  import pprint

import rlrom.app

from simglucose.simulation.sim_engine import batch_sim
from simglucose_init import PATIENT_NAMES, build_sim_obj

env   = "simglucose_not_really_env"
model = "offline_everything_was_actually_computed_before"

tester.env = env
tester.model_id = model

# creates one mock run to begin with

trace_idx = 0
trace = []
total_reward = 0
for i in range(10):
    time_step = i
    obs= [i*3.14]
    action = [0]
    next_obs = [(i+1)*3.14]
    reward = 1
    total_reward += reward
    done = False

    
    state = [time_step, obs, action, next_obs, reward, done]
    trace.append(state)


trace_idx = tester.add_trace(trace)
new_record = pd.DataFrame({'trace_idx':trace_idx, 
                           'env_name':tester.env_name,
                           'model_name': [tester.model_id],
                           'seed': [trace_idx], 'total_reward': [total_reward]})

if tester.evals is None:
    tester.evals = new_record
else:
    tester.evals = pd.concat([tester.evals, new_record])

print("trace",trace)
tester.signals_names = ["BG"]

spec_prompt = "alw BG[t] > 100"
plot_prompt = "_tr(0),BG"
[st, fig] = update_plot(spec_prompt,  plot_prompt)

with gr.Blocks(fill_height=True) as sg_gui:
    with gr.Row():
        textbox_bg_specs = gr.Textbox(label="Specifications",lines=5, interactive=True)         
        textbox_bg_plot_prompt = gr.Textbox(label="Plot layout",lines=5, interactive=True)            
    with gr.Row():
        button_bg_eval = gr.Button("Eval Specs")                   
        button_bg_plot = gr.Button("Update Plot")
    status_bg= gr.Textbox(label="Status", interactive=False)
    with gr.Tabs():
        with gr.Tab(label="Evaluations"):
            dataframe_bg_evals = gr.DataFrame(row_count= (8, 'dynamic'),value= tester.evals, interactive= False)
        with gr.Tab(label="Plot"):
            with gr.Group():
                fig_bg = gr.Plot()
    button_bg_plot.click(update_plot, [textbox_bg_specs, textbox_bg_plot_prompt], [status_bg, fig_bg])    
    button_bg_eval.click(eval_stl,    [textbox_bg_specs, dataframe_bg_evals], [status_bg, dataframe_bg_evals])
    
if __name__ == '__main__':
    sg_gui.launch()
    
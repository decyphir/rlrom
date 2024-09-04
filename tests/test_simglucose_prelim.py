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
new_record = pd.DataFrame({'trace_idx':trace_idx, 'env_name':tester.env_name,'model_name': [tester.model_id],'seed': [trace_idx], 'total_reward': [total_reward]})

if tester.evals is None:
    tester.evals = new_record
else:
    tester.evals = pd.concat([tester.evals, new_record])
       

print("trace",trace)
tester.signals_names = ["BG"]


with gr.Blocks(fill_height=True) as sg_gui:
    textbox_specs = gr.Textbox(label="Specifications",lines=10, interactive=True)         
    textbox_plot_prompt = gr.Textbox(label="Plot layout",lines=5, interactive=True)            
    with gr.Row():
            button_eval = gr.Button("Eval Specs")                   
            button_plot = gr.Button("Update Plot")
    dataframe_sg_evals = gr.DataFrame(row_count= (8, 'dynamic'),value= tester.evals,  interactive= False)
    with gr.Tabs():
        with gr.Tab(label="Evaluations"):
            dataframe_evals = gr.DataFrame(row_count= (8, 'dynamic'), interactive= False)
        with gr.Tab(label="Plot"):
            with gr.Group():
                fig = gr.Plot()
button_plot.click(update_plot, [textbox_specs,textbox_plot_prompt], [fig, textbox_status])    
button_eval.click(eval_stl, [textbox_specs, dataframe_evals], [textbox_status, dataframe_evals])
    




if __name__ == '__main__':
    #[specs.value, plot_prompt.value, status.value]  = callback_env(env)
    sg_gui.launch()
    
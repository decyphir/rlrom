import gradio as gr
from pprint import pprint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd

import rlrom
import rlrom.testers as testers
import rlrom.utils as hf

import stlrom

# gui state should be the tester
tester = testers.RLModelTester()
df_signals = pd.DataFrame(dtype=float)
df_robs = pd.DataFrame(dtype=float)
stl_driver = stlrom.STLDriver()

# function for run button
def run(env_name, repo_id, num_steps, seed, render_mode):
    
    if render_mode:
        render_mode = "human"
    else:
        render_mode = None
    
    try:
        
        model = hf.load_model(env_name, repo_id)
        if model is not None:
            tester = testers.RLModelTester(env_name, model, render_mode)   
        else:
            tester = testers.RLModelTester(env_name, None,  render_mode)

        tot_reward = tester.test_random(seed, num_steps)
        print('tot_reward:', tot_reward)
        # write signal in df_signals        
        tester.get_df_signals(df_signals)
        print(df_signals.head(5))
        status = "Test completed for " + env_name + " with model " + repo_id + " and reward: " + str(tot_reward)
    except Exception as e:
        status = "Error: " + str(e)
        return status
    return status

# add listener to env_dropdown
def update_models(env_name):
    try: 
        tester.env_name = env_name
        tester.create_env()
        _,models_ids = hf.find_models(env_name)
       
        # add "None" at the beginning of the list of models 
        models_ids = ["None"] + models_ids
        
        status = "Found " + str(len(models_ids)) + " models for " + env_name

        specs = tester.get_signal_string()
        specs = specs + """ 
        mu_r := reward[t]>0
        phi_alw := alw_[0, 100] mu_r
        phi_ev := ev_[0, 100] mu_r 
        """
        
        plot_prompt = ', '.join(tester.signals_names)
        plot_prompt += "\nmu_r, phi_alw, phi_ev"

        return [gr.Dropdown(
            choices=models_ids, value="None"), 
            specs,
            plot_prompt,
            status,
           ]    
    
    except Exception as e:
        status = "Error: " + str(e)
        return [gr.Dropdown(
            choices=["None"], value="None"), "","","",status] 
            
def update_plot(signals_plot_string):
    
    signals_layout = hf.get_layout_from_string(signals_plot_string)
    print(signals_layout)
    row_heights = [200 for _ in signals_layout]
    fig = make_subplots(rows=len(signals_layout),row_heights=row_heights, cols=1,shared_xaxes=True, vertical_spacing=0.02)
    fig.update_layout()
    status = "Plot all good."            
    for i, signal_list in enumerate(signals_layout):
        for signal in signal_list:
            if signal in df_signals.columns:
                fig.add_trace(go.Scatter(x=df_signals["time"], y=df_signals[signal], name=signal), row=i+1, col=1)
            elif signal in df_robs.columns:
                fig.add_trace(go.Scatter(x=df_robs["time"], y=df_robs[signal], name='rob('+signal+')'), row=i+1, col=1)
            else:
                status = "Warning: " + signal + " not found"
    
    return fig, status

def eval_stl(specs):
    try:
        stl_driver = stlrom.STLDriver()
        stl_driver.parse_string(specs)
        for idx in range(len(df_signals)):
            sample = df_signals.iloc[idx].values
            stl_driver.add_sample(sample)
        # list of formulas
        formulas = hf.get_formulas(specs)
        
        # clear df_robs without create a new one
        df_robs.drop(df_robs.index, inplace=True)
        # create the time  and formulas columns 
        df_robs["time"] = df_signals["time"]
        for formula in formulas:
            df_robs[formula] = 0

        for idx in range(len(df_signals)):    
            time = df_signals["time"][idx]
            values = []
            for idx_formula in range(len(formulas)):
                v = stl_driver.get_online_rob(formulas[idx_formula], time)[0]
                values.append(v)
            df_robs.loc[idx] = [time] + values
        status = str(df_robs.head(1))
    except Exception as e:
        status = "Error: " + str(e)
    return status

# create the layout
with gr.Blocks() as web_gui:
    with gr.Row():
        with gr.Column() as env_col:
            env_dropdown = gr.Dropdown(rlrom.supported_envs, label="Environment")
            models_dropdown = gr.Dropdown([], label="Model")
            with gr.Row():
                num_steps = gr.Number(value=100, label="Number of Steps")
                seed = gr.Number(value=1, label="Seed")
                render_mode = gr.Checkbox(label="Human Render Mode")
            run_button = gr.Button("Run")    
        with gr.Column() as txt_col:
            specs = gr.Textbox(label="STL Requirement(s)",lines=5, interactive=True)
            plot_prompt = gr.Textbox(label="Plot those",lines=2, interactive=True)            
            with gr.Row():
                button_eval = gr.Button("Eval STL Requirement(s)")                   
                button_plot = gr.Button("Update Plot")
    fig = gr.Plot()
    status = gr.Textbox(label="Status", interactive=False)    
    
    
    # callbacks    
    run_button.click(fn=run, inputs=[env_dropdown, models_dropdown, num_steps, seed, render_mode], outputs=status)
    env_dropdown.change(update_models, env_dropdown, [models_dropdown, specs, plot_prompt, status])
    button_plot.click(update_plot, [plot_prompt], [fig, status])    
    button_eval.click(eval_stl, [specs], status)

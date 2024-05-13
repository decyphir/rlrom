import gradio as gr
from pprint import pprint

import pandas as pd

import rlrom
import rlrom.testers as testers
import rlrom.utils as hf

import stlrom

# gui state should be the tester
tester = testers.RLModelTester()
df_signals = pd.DataFrame(dtype=float)
df_robs = pd.DataFrame()
stl_driver = stlrom.STLDriver()

# function for run button
def run(env_name, repo_id, num_steps, seed_list_str, render_mode):
    print('Entering run')
    print(tester.evals)
    
    if render_mode:
        render_mode = "human"
        lazy = False
    else:
        render_mode = None
        lazy = True
    
    try:
        model = hf.load_model(env_name, repo_id)    
        if tester.env_name != env_name:
            tester.reset()
            tester.env_name = env_name        
        tester.model = model
        tester.model_id = repo_id # shouldn't be necessary but            
        seed_list = hf.parse_integer_set_spec(seed_list_str)
        for seed in seed_list:
            tot_reward = tester.test_seed(seed, num_steps, render_mode=render_mode, lazy=lazy)
            print('seed:', seed, ' tot_reward:', tot_reward)
        # write signal in df_signals        
        tester.get_dataframe_from_trace(df_signals)
        #print(df_signals.head(5))
        status = "Test completed for " + env_name + " with model " + repo_id
        print(tester.evals.head(5))        
        return status, tester.evals

    except Exception as e:
        status = "Error: " + str(e)
        return status, None
    
# add listener to env_dropdown
def update_models(env_name):
    try: 
        tester.env_name = env_name
        tester.create_env()
        
        # remove all columns in df_signals
        df_signals.drop(df_signals.columns, axis=1, inplace=True)
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
        plot_prompt += """
        rho(mu_r), sat(mu_r), 
        rho(phi_alw),sat(phi_alw)
        rho(phi_ev), sat(phi_ev)
        """
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
            
def update_plot(specs, signals_plot_string):
    tester.specs = specs
    fig, status = tester.get_fig(signals_plot_string)
    return fig, status

def eval_stl(specs, df_evals):
    try:
        stl_driver = stlrom.STLDriver()
        stl_driver.parse_string(specs)
        for idx in range(len(df_signals)):
            sample = df_signals.iloc[idx].values
            stl_driver.add_sample(sample)
        # list of formulas
        tester.specs = specs
        formulas = hf.get_formulas(specs)

        for f in formulas:
            if f.startswith('phi'):
                tester.eval_spec(f)
        status = "STL evaluation completed"
        df_evals = tester.evals
    except Exception as e:
        status = "Error: " + str(e)
    return status, df_evals


# create the layout
with gr.Blocks(fill_height=True) as web_gui:
    with gr.Row():
        with gr.Column(scale=2) as env_col:
            env_dropdown = gr.Dropdown(rlrom.supported_envs, label="Environment")
            models_dropdown = gr.Dropdown([], label="Model")
            with gr.Row():
                num_steps = gr.Number(value=100, label="Number of Steps")
                seed_list = gr.Textbox(value=1, label="Seed(s)")
                render_mode = gr.Checkbox(label="Human Render Mode")
            run_button = gr.Button("Run")                    
        
        with gr.Column(scale=2) as txt_col:
            specs = gr.Textbox(label="STL Requirement(s)",lines=5, interactive=True)
            plot_prompt = gr.Textbox(label="Plot layout",lines=2, interactive=True)            
            with gr.Row():
                button_eval = gr.Button("Eval STL Requirement(s)")                   
                button_plot = gr.Button("Update Plot")
    with gr.Tabs():
        with gr.Tab(label="Evaluation"):
            table_evals = gr.DataFrame(row_count= (8, 'dynamic'))
        with gr.Tab(label="Plot"):
            with gr.Group():
                fig = gr.Plot(scale=1000)
    
    status = gr.Textbox(label="Status", interactive=False)    
        
    # callbacks    
    run_button.click(fn=run, inputs=[env_dropdown, models_dropdown, num_steps, seed_list, render_mode], outputs=[status, table_evals])
    env_dropdown.change(update_models, env_dropdown, [models_dropdown, specs, plot_prompt, status])
    button_plot.click(update_plot, [specs,plot_prompt], [fig, status])    
    button_eval.click(eval_stl, [specs, table_evals], [status, table_evals])

import gradio as gr
from pprint import pprint

import pandas as pd

import rlrom
import rlrom.testers as testers
import rlrom.utils as hf

import stlrom


# gui state should be the tester
tester = testers.RLModelTester()
reset_confirm = False
local_models_list = []

default_specs = """
positive_reward := reward[t]>0
phi_ev_pos_rew := ev_[0, 100] positive_reward"""

# default plots: first select trace idx 0, then plot reward and phi
default_plots = """
_tr(0)
positive_reward, sat(positive_reward)
phi_ev_pos_rew, sat(phi_ev_pos_rew)"""

cfg_envs = rlrom.cfg_envs # load default environment configurations


def update_plot(specs, signals_plot_string):
    tester.specs = specs
    fig, status = tester.get_fig(signals_plot_string)    
    return status, fig

def eval_stl(specs, df_evals):
    try:
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
    with gr.Tabs():
        with gr.Tab(label="Configuration and Model Testing"):
        #with gr.Column as env_col:
            with gr.Row():
                dropdown_cfg = gr.Dropdown(['']+rlrom.supported_envs, label="Configuration", scale=2)
                dropdown_source = gr.Dropdown(['Random','Manual (Random if not available)', 'Local', 'Hugging Face'], 
                                              label="Model Source")
            
            button_upload = gr.UploadButton("Upload a model file", file_count="single") 
            dropdown_models = gr.Dropdown([], label="Model")
                
            with gr.Row():
                num_steps = gr.Number(value=100, label="Number of Steps")
                seed_list = gr.Textbox(value=0, label="Seed(s)")
                with gr.Column():
                    checkbox_render = gr.Checkbox(label="Human Render Mode")            
                    checkbox_lazy = gr.Checkbox(label="Lazy (don't recompute seed)")            
        with gr.Tab(label="Specifications and Plotting"):
            textbox_specs = gr.Textbox(label="Specifications",lines=10, interactive=True)
            textbox_plot_prompt = gr.Textbox(label="Plot layout",lines=5, interactive=True)            
    with gr.Row():
            button_run = gr.Button("Run")
            button_eval = gr.Button("Eval Specs")                   
            button_plot = gr.Button("Update Plot")
            button_reset = gr.Button("Reset")
                
    textbox_status = gr.Textbox(label="Status", interactive=False)    

    with gr.Tabs():
        with gr.Tab(label="Evaluations"):
            dataframe_evals = gr.DataFrame(row_count= (8, 'dynamic'), interactive= False)
        with gr.Tab(label="Plot"):
            with gr.Group():
                fig = gr.Plot()
    
    
    # Defines all callbacks             
    
    def callback_reset_evals():
        global reset_confirm
        global tester    
        if reset_confirm is False:
            reset_confirm = True    
            gr.Warning('This will remove all evaluations done so far. Are you sure ?')
            return 'Click again to confirm.', tester.evals
        else:
            reset_confirm = False    
            gr.Warning('Data reset. Please refresh the page to start again.')
            tester =  testers.RLModelTester()
            return 'Good as new', tester.evals

    def callback_source(env_name, source_type):  
        models_list  = ['None']
        status = "No models found for " + env_name

        try:
            if source_type == 'Local' and len(local_models_list) > 0:
                models_list += local_models_list               
                status = "Found " + str(len(local_models_list)) + " local models for " + env_name
            elif source_type == 'Hugging Face':
                _,hf_models = hf.find_models(env_name)       
                models_list += hf_models
                status = "Found " + str(len(hf_models)) + " models for " + env_name
            elif source_type == 'Manual (Random if not available)':
                status = "Manual mode selection. Random model will be selected if not available"

            dropdown=   gr.Dropdown(choices=models_list, value="None")

        except Exception as e:
            status = "Error: " + str(e)
            dropdown = gr.Dropdown(choices=["None"], value="None")

        return [status, dropdown]

    def callback_env(cfg_name):
        global tester
        try:         
            cfg = cfg_envs[cfg_name]
            tester = rlrom.RLModelTester(cfg)
            
            if 'specs' in cfg:
                specs = cfg['specs']
            else:
                specs = tester.get_signal_string()
                specs = specs + default_specs

            if 'plots' in cfg:
                plot_prompt = cfg['plots']
            else:
                plot_prompt = ', '.join(tester.signals_names)
                plot_prompt += default_plots

            status = "Configuration " + cfg + " loaded"        
            return [
                specs,
                plot_prompt,
                status,
               ]    

        except Exception as e:
            status = "Error: " + str(e)
            return [gr.Dropdown(
                choices=["None"], value="None"), "","","",status] 

    def callback_upload(file):
        global local_models_list
        local_models_list += [file.name]
        status = "Added " + file.name
        dropdown_models = gr.Dropdown(choices=local_models_list, value=file.name)

        return  status,dropdown_models

    # function for run button
    def run(env_name,model_src, model_name, num_steps, seed_list_str, render_mode, lazy_mode):
        print('Entering run')
        print(tester.evals)
        
        if render_mode: # force non lazy if human rendering
            render_mode = "human"
            lazy = False
        else:
            render_mode = None
            lazy = lazy_mode
        
        try:
            # update environment if necessary
            if tester.env_name != env_name:
                tester.reset()
                tester.env_name = env_name        
    
            seed_list = hf.parse_integer_set_spec(seed_list_str)
    
            # runs the tests
            for seed in seed_list:
                model = None
                if model_src == 'Local':
                    model = hf.load_model(env_name=env_name, filename= model_name)    
                elif model_src == 'Manual (Random if not available)':
                    model_name = 'Manual'    
                elif model_src == 'Hugging Face':
                    model = hf.load_model(env_name=env_name, repo_id=model_name)
                tester.model = model
                tester.model_id = model_name # shouldn't be necessary but            
    
                tot_reward = tester.test_seed(seed, num_steps, render_mode=render_mode, lazy=lazy_mode)
                print('seed:', seed, ' tot_reward:', tot_reward)
    
            status = "Test completed for " + env_name + " with model " + model_name
            return status, tester.evals
    
        except Exception as e:
            status = "Error: " + str(e)
            return status, None


    # callbacks    
    button_run.click(fn=run, inputs=[dropdown_cfg, dropdown_source, dropdown_models, num_steps, seed_list, checkbox_render, checkbox_lazy], outputs=[textbox_status, dataframe_evals])
    button_upload.upload(callback_upload, button_upload, [textbox_status, dropdown_models])
    dropdown_source.change(callback_source, [dropdown_cfg, dropdown_source], [textbox_status, dropdown_models])  
    dropdown_cfg.change(callback_env, dropdown_cfg, [textbox_specs, textbox_plot_prompt, textbox_status])    
    button_plot.click(update_plot, [textbox_specs,textbox_plot_prompt], [textbox_status, fig])    
    button_eval.click(eval_stl, [textbox_specs, dataframe_evals], [textbox_status, dataframe_evals])
    button_reset.click(callback_reset_evals, [], [textbox_status,dataframe_evals])

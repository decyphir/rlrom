import gradio as gr


from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure



def greet(name):
    # list will contain from 0 to 6
    currentList = list(range(7))
    
    # y0 is equals to x no change
    List1 = currentList
    
    # y1 square root of x
    List2 = [i**0.5 for i in currentList]
    
    # y2 square  of x
    List3 = [i**2 for i in currentList]
    
    colors = itertools.cycle(inferno(numLines))# create a color iterator 

    # now creating plots f1,f2,f3
    f1 = figure(height=200)
    f1.line(currentList, List1, legend_label='sig1')
    f1.line(currentList, List3, legend_label='sig3')
 
    f2 = figure(height=200)
    f2.line(currentList, List2, legend_label='sig2')
    
    f = gridplot([[f1],[f2]],sizing_mode='stretch_width')


    return f

with gr.Blocks() as demo:
    with gr.Column():
        name = gr.Textbox(label="Names")
        output = gr.Plot(label="Output Box")
        greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")

# run the app
if __name__ == "__main__":
    demo.launch()
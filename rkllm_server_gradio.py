import sys
import resource
import gradio as gr
import psutil
from ctypes_bindings import *
from model_class import *
from mesh_utils import *

# Set environment variables
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "8080"
os.environ["RKLLM_LOG_LEVEL"] = "1"
# Set resource limit
resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

if __name__ == "__main__":
    # Helper function to define initializing model before class is declared
    # Without this, you would need to initialize the class before you select the model
    def initialize_model(model):
        global rkllm_model
        # Have to unload previous model in single-threaded mode
        try:
            rkllm_model.release()
        except:
            print("No model loaded! Continuing with initialization...")
        # Initialize RKLLM model
        init_msg = "=========INITIALIZING==========="
        print(init_msg)
        sys.stdout.flush()
        rkllm_model = RKLLMLoaderClass(model=model)
        model_init = f"RKLLM Model, {rkllm_model.model_name} has been initialized successfully！"
        print(model_init)
        complete_init = "=============================="
        print(complete_init)
        output = [[f"<h4 style=\"text-align:center;\">{model_init}\n</h4>", None]]
        sys.stdout.flush()
        return output 

    # Wrapper functions so the class can be loaded dynamically
    def get_user_input(user_message, history):
        try:
            prompt, history = rkllm_model.get_user_input(user_message, history)
            return "", history
        except RuntimeError as e:
            print(f"ERROR: {e}")

    def get_RKLLM_output(history):
        try:
            yield from rkllm_model.get_RKLLM_output(history)
            print(history)
        except RuntimeError as e:
            print(f"ERROR: {e}")
        return history
        
    def get_ram_usage():
        # Get memory usage details
        memory = psutil.virtual_memory()
        total_ram = memory.total / (1024 ** 3)  # Convert from bytes to GB
        percent = memory.percent
        used_ram = percent * total_ram / 100
        ram_usage = f"{percent}%; {used_ram:.2f} GB / {total_ram:.2f} GB"
        return ram_usage

    # Create a Gradio interface
    with gr.Blocks(title="Chat with RKLLM") as chatRKLLM:
        available_models = available_models()
        gr.Markdown("<div align='center'><font size='10'> Definitely Not Skynet </font></div>")
        with gr.Tabs():
            with gr.TabItem("Select Model"):
                model_dropdown = gr.Dropdown(choices=available_models, label="Select Model", value="Flush")
                statusBox = gr.Chatbot(height=100)
                model_dropdown.input(initialize_model, [model_dropdown], [statusBox])
            with gr.TabItem("Txt2Txt"):
                msg = gr.Textbox(placeholder="Please input your question here...", label="Send a message")
                rkllmServer = gr.Chatbot()
                msg.submit(get_user_input, [msg, rkllmServer], [msg, rkllmServer], queue=True).then(get_RKLLM_output, rkllmServer, rkllmServer)
            with gr.TabItem("Txt2Mesh"):
                with gr.Row():    
                    with gr.Column(scale=3):
                        msg = gr.Textbox(placeholder="Please input your question here...", label="inputTextBox")
                        statusBox = gr.Chatbot()
                        # model_dropdown.input(initialize_model, [model_dropdown], [statusBox])
                        msg.submit(get_user_input, [msg, statusBox], [msg, statusBox], queue=True).then(get_RKLLM_output, statusBox, statusBox)
                        clear = gr.Button("Clear")
                        clear.click(lambda: None, None, rkllmServer, queue=False)
                    with gr.Column(scale=2):
                        # Add the text box for 3D mesh input and button
                        mesh_input = gr.Textbox(
                            label="3D Mesh Input",
                            placeholder="Paste your 3D mesh in OBJ format here...",
                            lines=5,
                        )
                        visualize_button = gr.Button("Visualize 3D Mesh")
                        output_model = gr.Model3D(
                                    label="3D Mesh Visualization",
                                    interactive=False,
                                )
                        # Link the button to the visualization function
                        visualize_button.click(
                            fn=apply_gradient_color,
                            inputs=[mesh_input],
                            outputs=[output_model]
                            )
        print("\nNo model loaded yet!\n")

        # Add a button to display RAM usage
        def display_ram_usage():
            ram_usage = get_ram_usage()  # Get the formatted RAM usage
            return "Click to refresh " + ram_usage  # This will be returned to the RAM display Textbox

        ram_button = gr.Button("Click to display RAM Usage")
        ram_button.click(display_ram_usage, outputs=ram_button)

    # Enable the event queue system.
    chatRKLLM.queue()
    # Start the Gradio application.
    chatRKLLM.launch()

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")

#!/usr/bin/env python
# encoding: utf-8

import timm
import gradio as gr
from PIL import Image
import traceback
import re
import torch
import argparse
from transformers import AutoModel, AutoTokenizer

# Suppress FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# README, How to run demo on different devices
# For CPU usage, you can simply run:
# python app.py

# Argparser
parser = argparse.ArgumentParser(description='Demo Application Configuration')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu'], help='Device to run the model on. Currently only "cpu" is supported.')
parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32'], help='Data type for model computations. "fp32" is standard for CPU.')
args = parser.parse_args()

device = args.device

# Since we're using CPU, set dtype to float32
if args.dtype == 'fp32':
    dtype = torch.float32
else:
    dtype = torch.float32  # Fallback to float32 if an unsupported dtype is somehow passed

# Load model
model_path = 'openbmb/MiniCPM-V-2'

try:
    print("Loading model...")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    exit(1)

model.eval()

ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-V 2.0'

# Define UI components parameters
form_radio = {
    'choices': ['Beam Search', 'Sampling'],
    'value': 'Sampling',
    'interactive': True,
    'label': 'Decode Type'
}

# Beam Search Parameters
num_beams_slider = {
    'minimum': 1,  # Changed minimum from 0 to 1 as 0 beams doesn't make sense
    'maximum': 10,  # Increased maximum for more flexibility
    'value': 3,
    'step': 1,
    'interactive': True,
    'label': 'Num Beams'
}
repetition_penalty_slider = {
    'minimum': 0.5,  # Changed minimum to a reasonable value
    'maximum': 3.0,
    'value': 1.2,
    'step': 0.01,
    'interactive': True,
    'label': 'Repetition Penalty'
}

# Sampling Parameters
repetition_penalty_slider2 = {
    'minimum': 0.5,
    'maximum': 3.0,
    'value': 1.05,
    'step': 0.01,
    'interactive': True,
    'label': 'Repetition Penalty'
}
max_new_tokens_slider = {
    'minimum': 1,
    'maximum': 4096,
    'value': 1024,
    'step': 1,
    'interactive': True,
    'label': 'Max New Tokens'
}

top_p_slider = {
    'minimum': 0.1,  # Avoid extreme low values
    'maximum': 1.0,
    'value': 0.8,
    'step': 0.05,
    'interactive': True,
    'label': 'Top P'
}
top_k_slider = {
    'minimum': 10,  # Avoid extreme low values
    'maximum': 200,
    'value': 100,
    'step': 1,
    'interactive': True,
    'label': 'Top K'
}
temperature_slider = {
    'minimum': 0.1,  # Avoid extreme low values
    'maximum': 2.0,
    'value': 0.7,
    'step': 0.05,
    'interactive': True,
    'label': 'Temperature'
}

def create_component(params, comp='Slider'):
    """
    Utility function to create Gradio UI components based on parameters.
    """
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )

def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
    """
    Function to handle the chat interaction.
    """
    print("Entering chat function...")
    default_params = {"num_beams": 3, "repetition_penalty": 1.2, "max_new_tokens": 1024}
    if params is None:
        params = default_params
    if img is None:
        return -1, "Error, invalid image, please upload a new image", None, None
    try:
        image = img.convert('RGB')
        answer, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            **params
        )
        # Clean up the answer text
        res = re.sub(r'(<box>.*</box>)', '', answer)
        res = res.replace('<ref>', '').replace('</ref>', '').replace('<box>', '').replace('</box>', '')
        answer = res
        return -1, answer, None, None
    except Exception as err:
        print(err)
        traceback.print_exc()
        return -1, ERROR_MSG, None, None

def upload_img(image, _chatbot, _app_session):
    """
    Function to handle image uploads.
    """
    print("Uploading image...")
    try:
        image = Image.fromarray(image)
        _app_session['sts'] = None
        _app_session['ctx'] = []
        _app_session['img'] = image
        _chatbot.append(('', 'Image uploaded successfully, I am ready to take up your queries'))
        print("Image uploaded successfully.")
        return _chatbot, _app_session
    except Exception as e:
        print(f"Error uploading image: {e}")
        traceback.print_exc()
        return _chatbot, _app_session

def respond(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature):
    """
    Function to handle user input and generate responses.
    """
    print("Respond function called.")
    if _app_cfg.get('ctx', None) is None:
        _chat_bot.append((_question, 'Please upload an image to detect'))
        return '', _chat_bot, _app_cfg

    _context = _app_cfg['ctx'].copy()
    if _context:
        _context.append({"role": "user", "content": _question})
    else:
        _context = [{"role": "user", "content": _question}]
    print('<User>:', _question)

    if params_form == 'Beam Search':
        params = {
            'sampling': False,
            'num_beams': num_beams,
            'repetition_penalty': repetition_penalty,
            "max_new_tokens": 896 
        }
    else:
        params = {
            'sampling': True,
            'top_p': top_p,
            'top_k': top_k,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty_2,
            "max_new_tokens": 896 
        }
    code, _answer, _, sts = chat(_app_cfg['img'], _context, None, params)
    print('<Assistant>:', _answer)

    _context.append({"role": "assistant", "content": _answer}) 
    _chat_bot.append((_question, _answer))
    if code == 0:
        _app_cfg['ctx'] = _context
        _app_cfg['sts'] = sts
    return '', _chat_bot, _app_cfg

def regenerate_button_clicked(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature):
    """
    Function to handle the regeneration of the last assistant response.
    """
    print("Regenerate button clicked.")
    if len(_chat_bot) <= 1:
        _chat_bot.append(('Regenerate', 'No question for regeneration.'))
        return '', _chat_bot, _app_cfg
    elif _chat_bot[-1][0] == 'Regenerate':
        return '', _chat_bot, _app_cfg
    else:
        _question = _chat_bot[-1][0]
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
    return respond(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature)

# Building the Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            # Decode Type Selection
            params_form = create_component(form_radio, comp='Radio')
            
            # Beam Search Settings
            with gr.Accordion("Beam Search"):
                num_beams = create_component(num_beams_slider)
                repetition_penalty = create_component(repetition_penalty_slider)
            
            # Sampling Settings
            with gr.Accordion("Sampling"):
                top_p = create_component(top_p_slider)
                top_k = create_component(top_k_slider)
                temperature = create_component(temperature_slider)
                repetition_penalty_2 = create_component(repetition_penalty_slider2)
            
            # Regenerate Button
            regenerate = create_component({'value': 'Regenerate'}, comp='Button')
        
        with gr.Column(scale=3, min_width=500):
            # Application State
            app_session = gr.State({'sts': None, 'ctx': None, 'img': None})
            
            # Image Upload Component
            bt_pic = gr.Image(label="Upload an image to start")
            
            # Chatbot Display
            chat_bot = gr.Chatbot(label="Ask anything about the image")
            
            # Text Input for User Messages
            txt_message = gr.Textbox(label="Input text")
            
            # Define Actions
            regenerate.click(
                regenerate_button_clicked,
                [
                    txt_message, 
                    chat_bot, 
                    app_session, 
                    params_form, 
                    num_beams, 
                    repetition_penalty, 
                    repetition_penalty_2, 
                    top_p, 
                    top_k, 
                    temperature
                ],
                [txt_message, chat_bot, app_session]
            )
            
            txt_message.submit(
                respond, 
                [
                    txt_message, 
                    chat_bot, 
                    app_session, 
                    params_form, 
                    num_beams, 
                    repetition_penalty, 
                    repetition_penalty_2, 
                    top_p, 
                    top_k, 
                    temperature
                ], 
                [txt_message, chat_bot, app_session]
            )
            
            bt_pic.upload(
                lambda: None, 
                None, 
                chat_bot, 
                queue=False
            ).then(
                upload_img, 
                inputs=[bt_pic, chat_bot, app_session], 
                outputs=[chat_bot, app_session]
            )

# Launch the Gradio App with share=True for testing
demo.launch(share=True, debug=True)
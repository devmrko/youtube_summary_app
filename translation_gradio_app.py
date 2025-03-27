import gradio as gr
import requests
import json
import sys
import os
import argparse
import ast

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'reconfigure') else None

# API Endpoint
API_ENDPOINT = "http://193.123.248.120:5000/translate"

# Language options and codes
LANGUAGES = {
    "English": "en",
    "Korean": "ko"
}

# Reverse mapping for displaying language names
LANGUAGE_NAMES = {code: name for name, code in LANGUAGES.items()}

def translate_text(input_text, source_lang, target_lang):
    """
    Send translation request to the API and process the response
    """
    if not input_text.strip():
        return "Please enter some text to translate."
    
    # Prepare the request payload
    payload = {
        "text": input_text,
        "source_lang": LANGUAGES[source_lang],
        "target_lang": LANGUAGES[target_lang]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        # Send the request to the API
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Check for error in response
            if 'error' in result:
                return f"Translation Error: {result['error']}"
            
            # Extract the translated text
            if isinstance(result, dict):
                translated_text = result.get('translated_text', 'No translation found')
            else:
                translated_text = result
            
            # Handle different response formats
            if isinstance(translated_text, list):
                # If it's a list of dictionaries with translation_text
                if translated_text and isinstance(translated_text[0], dict) and 'translation_text' in translated_text[0]:
                    return ' '.join([item['translation_text'] for item in translated_text])
                # If it's a list of strings
                elif translated_text and all(isinstance(item, str) for item in translated_text):
                    return ' '.join(translated_text['translation_text'])
                # Other list type, convert to string
                else:
                    return str(translated_text)
            # If it's already a string
            elif isinstance(translated_text, str):
                return translated_text
            # Other type, convert to string
            else:
                return str(translated_text)
        else:
            return f"Error: API returned status code {response.status_code}\n{response.text}"
    
    except Exception as e:
        return f"Error connecting to translation API: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Translation App", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Translation App")
    gr.Markdown("Translate text between English, Korean, and Japanese")
    
    with gr.Row():
        with gr.Column():
            # Source language and text
            source_lang = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="English",
                label="Source Language"
            )
            
            source_text = gr.Textbox(
                lines=10,
                placeholder="Enter text to translate...",
                label="Source Text"
            )
        
        with gr.Column():
            # Target language and translated text
            target_lang = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="Korean",
                label="Target Language"
            )
            
            translated_text = gr.Textbox(
                lines=10,
                label="Translated Text",
                interactive=False
            )
    
    # Button and functionality
    with gr.Row():
        translate_btn = gr.Button("Translate", variant="primary")
        clear_btn = gr.Button("Clear")
        swap_btn = gr.Button("Swap Languages")
    
    # Set up button actions
    translate_btn.click(
        fn=translate_text,
        inputs=[source_text, source_lang, target_lang],
        outputs=translated_text
    )
    
    clear_btn.click(
        fn=lambda: ["", "English", "Korean", ""],
        inputs=None,
        outputs=[source_text, source_lang, target_lang, translated_text]
    )
    
    def swap_languages(src_lang, tgt_lang):
        return tgt_lang, src_lang
    
    swap_btn.click(
        fn=swap_languages,
        inputs=[source_lang, target_lang],
        outputs=[source_lang, target_lang]
    )
    
    # Add examples
    with gr.Accordion("Examples", open=False):
        gr.Examples(
            examples=[
                ["Hello, how are you today?", "English", "Korean"],
                ["안녕하세요, 오늘 기분이 어떠세요?", "Korean", "English"]
            ],
            inputs=[source_text, source_lang, target_lang],
            outputs=translated_text,
            fn=translate_text
        )
    
    # Information about the app
    with gr.Accordion("About", open=False):
        gr.Markdown(f"""
        ## About this App
        
        This translation app uses a custom translation API built with NLLB models. It supports:
        
        - English (en)
        - Korean (ko)
        
        The application handles both short phrases and longer texts.
        
        **Current API Endpoint:** {API_ENDPOINT}
        """)

# Launch the app
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Translation Gradio App")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name/host")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()
    
    # Launch with specified parameters
    app.launch(
        server_name=args.server_name, 
        server_port=args.server_port, 
        share=args.share
    ) 
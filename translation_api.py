from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import os
from functools import lru_cache
import re
import ast

app = Flask(__name__)
# Configure Flask to not escape non-ASCII characters
app.config['JSON_AS_ASCII'] = False
app.json.ensure_ascii = False

# Language code mapping for NLLB model
LANG_MAP = {
    'en': 'eng_Latn',  # English
    'ko': 'kor_Hang',  # Korean
    'ja': 'jpn_Jpan'   # Japanese
}

# Models for different translation directions
MODELS = {
    'en2ko': 'NHNDQ/nllb-finetuned-en2ko',
    'ko2en': 'NHNDQ/nllb-finetuned-ko2en',  # Using base NLLB model
    'ja2ko': 'sappho192/aihub-ja-ko-translator',  # Based on reference
    'ko2ja': 'facebook/nllb-200-distilled-600M'#'facebook/nllb-200-distilled-600M'   # Using base NLLB model
}

# Default lengths for translations (increased to handle longer texts)
DEFAULT_MAX_TOKEN = 512
DEFAULT_MAX_LENGTH_ENGLISH = 180 #1300 # equivalent to 512 English tokens
DEFAULT_MAX_LENGTH_KOREAN = 180#800 # equivalent to 512 Korean tokens   
DEFAULT_MAX_LENGTH_JAPANESE = 180#700 # equivalent to 512 Japanese token

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'CUDA (GPU)' if device == 0 else 'CPU'}")

# Cache the translator instances to avoid reloading models
@lru_cache(maxsize=4)
def get_translator(source_lang, target_lang):
    """Get or create a translator for the specified language pair"""
    direction = f"{source_lang}2{target_lang}"
    
    if direction in MODELS:
        model_name = MODELS[direction]
        print(f"Loading model {model_name} for {direction}...")
        
        # Special case for Japanese to Korean model which might use a different architecture
        if direction == 'ja2ko':
            from transformers import EncoderDecoderModel, BertJapaneseTokenizer, PreTrainedTokenizerFast
            
            # This setup is based on translate_ja_2_ko_by_NLLB.py
            encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
            decoder_model_name = "skt/kogpt2-base-v2"
            
            src_tokenizer = BertJapaneseTokenizer.from_pretrained(encoder_model_name)
            trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(decoder_model_name)
            
            model = EncoderDecoderModel.from_pretrained(model_name)
            
            def translate_ja_ko(text, max_length=DEFAULT_MAX_LENGTH_JAPANESE):
                embeddings = src_tokenizer(text, return_attention_mask=False, 
                                        return_token_type_ids=False, return_tensors='pt')
                embeddings = {k: v for k, v in embeddings.items()}
                # Set minimum length to avoid early cutoff and high max length
                output = model.generate(
                    **embeddings, 
                    max_length=max_length,
                    min_length=int(len(text) * 0.5),  # Ensure minimum reasonable length
                    no_repeat_ngram_size=3,  # Avoid repetition
                    num_beams=5  # Increase beam search for better quality
                )[0, 1:-1]
                translated = trg_tokenizer.decode(output.cpu())
                return translated
            
            return translate_ja_ko
        else:
            # Standard NLLB model
            return pipeline('translation', 
                          model=model_name, 
                          device=device, 
                          src_lang=LANG_MAP[source_lang],
                          tgt_lang=LANG_MAP[target_lang])
    else:
        raise ValueError(f"Unsupported translation direction: {direction}")

def translate_text(text, source_lang, target_lang, max_token=None):
    """Translate text from source language to target language"""
    translator = get_translator(source_lang, target_lang)
    
    # Handle different translator types
    if callable(translator) and not hasattr(translator, '__self__'):
        # This is the custom ja2ko function
        return translator(text)
    else:
        # This is a transformers pipeline - use additional parameters for better control
        output = translator(
            text
        )
        
        return str(output)
    
def extract_translation_text(text_repr):
    """Safely extract translation text from various string representations"""
    import ast
    try:
        # Try to parse the string representation
        parsed = ast.literal_eval(text_repr)
        
        # Handle different structures
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and 'translation_text' in parsed[0]:
            return parsed[0]['translation_text']
        elif isinstance(parsed, dict) and 'translation_text' in parsed:
            return parsed['translation_text']
        else:
            # If structure is not as expected, return the original string
            return text_repr
    except (ValueError, SyntaxError):
        # If parsing fails, return the original string
        return text_repr

@app.route('/translate', methods=['POST'])
def translate():
    """API endpoint for translation"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Get parameters from request
    text = data.get('text')
    source_lang = data.get('source_lang', 'en').lower()
    target_lang = data.get('target_lang', 'ko').lower()

    # Set max length based on source language
    if source_lang == 'ja':
        max_length = data.get('max_length', DEFAULT_MAX_LENGTH_JAPANESE)
    elif source_lang == 'ko':
        max_length = data.get('max_length', DEFAULT_MAX_LENGTH_KOREAN)
    elif source_lang == 'en':
        max_length = data.get('max_length', DEFAULT_MAX_LENGTH_ENGLISH)
    else:
        max_length = data.get('max_length', 500) # for other languages, default to 500
    
    # Validate parameters
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if source_lang not in LANG_MAP:
        return jsonify({'error': f'Unsupported source language: {source_lang}'}), 400
    
    if target_lang not in LANG_MAP:
        return jsonify({'error': f'Unsupported target language: {target_lang}'}), 400
    
    # Check if this translation direction is supported
    direction = f"{source_lang}2{target_lang}"
    if direction not in MODELS:
        return jsonify({'error': f'Unsupported translation direction: {direction}'}), 400
    
    try:
        # For debugging
        print(f"Translating text of length {len(text)}, max_output_length set to {max_length}")
        
        # Split very long text into paragraphs if needed
        if len(text) > max_length:
            # Split by newlines first, then ensure each paragraph ends with punctuation
            paragraphs = []
            for p in text.split('\n'):
                if p.strip():
                    # Process the paragraph to add sentence endings where needed
                    p = p.strip()
                    
                    # Split into sentences first (to handle multiple sentences in one paragraph)
                    sentences = re.split(r'(?<=[.!?。！？])\s*', p)
                    
                    # Process each sentence
                    processed_sentences = []
                    for s in sentences:
                        s = s.strip()
                        if s:
                            # Add period if needed
                            if not s[-1] in '.!?。！？':
                                s = s + '. '  # Add period and space
                            processed_sentences.append(s)
                    
                    for x in processed_sentences:
                        paragraphs.append(x)
            
            if len(paragraphs) > 1:
                translated_paragraphs = []
                for para in paragraphs:
                    if para.strip():
                        try:
                            translated_para = translate_text(para, source_lang, target_lang, max_length)
                            # Ensure translated paragraph is a string
                            if not isinstance(translated_para, str):
                                translated_para = str(translated_para)
                            translated_paragraphs.append(translated_para)
                        except Exception as e:
                            print(f"Error translating paragraph: {str(e)}")
                            translated_paragraphs.append(f"[Translation Error]")
                
                # Ensure all elements in translated_paragraphs are strings
                translated_paragraphs = [str(p) if p is not None else "" for p in translated_paragraphs]

                translated_text = ''
                for x in translated_paragraphs:
                    translated_text += extract_translation_text(x) + ' '
                
            else:
                # Just one long paragraph
                try:
                    translated_text = translate_text(text, source_lang, target_lang, max_length)
                    if not isinstance(translated_text, str):
                        translated_text = str(translated_text)
                except Exception as e:
                    print(f"Error translating single paragraph: {str(e)}")
                    translated_text = f"[Translation Error: {str(e)}]"
        else:
            # Normal length text
            try:
                translated_text = translate_text(text, source_lang, target_lang, max_length)
                if not isinstance(translated_text, str):
                    translated_text = str(translated_text)
            except Exception as e:
                print(f"Error translating text: {str(e)}")
                translated_text = f"[Translation Error: {str(e)}]"
        
        # Return the result
        return jsonify({
            'source_lang': source_lang,
            'target_lang': target_lang,
            'source_text': text,
            'translated_text': translated_text
        })
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/supported_languages', methods=['GET'])
def supported_languages():
    """Return list of supported languages and translation directions"""
    supported_directions = list(MODELS.keys())
    supported_langs = list(LANG_MAP.keys())
    
    return jsonify({
        'supported_languages': supported_langs,
        'supported_directions': supported_directions
    })

if __name__ == '__main__':
    try:
        # Get port from environment or use default
        port = int(os.environ.get('PORT', 5000))
        
        # Try to use waitress (more stable on Windows) if available
        try:
            from waitress import serve
            print(f"Starting server with Waitress on http://0.0.0.0:{port}")
            serve(app, host='0.0.0.0', port=port)
        except ImportError:
            # Fall back to Flask's built-in server with debug disabled to avoid reloader issues
            print(f"Waitress not available, using Flask's built-in server on http://0.0.0.0:{port}")
            print("Consider installing waitress for better stability: pip install waitress")
            app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\nServer shutdown requested. Exiting...")
    except Exception as e:
        print(f"Error starting server: {e}") 
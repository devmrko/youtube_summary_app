import oci
from pathlib import Path
import sys
import os
import json
from datetime import datetime

# OCI Configuration
CONFIG_PATH = os.path.expanduser(os.getenv("OCI_CONFIG_PATH"))
KEY_PATH = os.path.expanduser(os.getenv("OCI_KEY_PATH"))
RESULTS_DIR = "results" # Results directory

# Translation parameters
SOURCE_LANG = "en" # Source language
TARGET_LANGS = ["ko", "es", "fr"]  # Korean, Spanish, French
TEST_TEXTS = [
    "Hello, how are you today?",
    "The weather is beautiful.",
    "Artificial Intelligence is transforming the world."
]

# Sentiment analysis parameters
SENTIMENT_TEXTS = [
    "I really love this product, it's amazing!",
    "The service was terrible and I'm very disappointed.",
    "The weather is okay today, nothing special."
]

# Verify OCI configuration files and settings  
def verify_config():
    """Verify OCI configuration files and settings."""
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: OCI config file not found at {CONFIG_PATH}")
        print("\nPlease create the config file with the following format:")
        print("""
[DEFAULT]
user=ocid1.user.oc1..aaaaaaaaxb5bqep45pgno24kivw3ecj6yotxrjqoz4p7a3zxgt6occu6pf2a
fingerprint=63:8f:6c:bd:f8:65:b4:36:8c:b0:03:2c:55:2c:d9:2c
key_file=C:/Users/jmko7/.oci/sessions/joungminkoaws/oci_api_key.pem
tenancy=ocid1.tenancy.oc1..aaaaaaaa2wpfjruuzputbhuzhz2cbwnizjgdyg5q2xqf6nlzm66wayav7a2a
region=us-ashburn-1""")
        return False
        
    if not os.path.exists(KEY_PATH): # Check if the API key file exists
        print(f"Error: API key file not found at {KEY_PATH}") # Print the error if the API key file does not exist
        return False
        
    try:
        config = oci.config.from_file()
        required_keys = ['user', 'fingerprint', 'key_file', 'tenancy', 'region'] # Required keys
        missing_keys = [key for key in required_keys if key not in config] # Check if the required keys are present in the config
        
        if missing_keys: # If the required keys are not present in the config
            print(f"Error: Missing required configuration keys: {', '.join(missing_keys)}") # Print the error if the required keys are not present in the config
            return False
            
        return True
    except Exception as e:
        print(f"Error parsing config file: {str(e)}") # Print the error if the config file is not parsed   
        return False

# Initialize the translation client
def init_translation_client(config):
    """Initialize the translation client."""
    return oci.ai_language.AIServiceLanguageClient(config)

# Create translation request details
def create_translation_details(text, target_lang):
    """Create translation request details."""
    return oci.ai_language.models.BatchLanguageTranslationDetails(
        documents=[
            oci.ai_language.models.TextDocument(
                key="1",
                text=text,
                language_code=SOURCE_LANG
            )
        ],
        target_language_code=target_lang
    )

# Test the translation service 
def test_translation():
    """Test the translation service."""
    if not verify_config():
        return False
        
    try:
        # Get the config        
        config = oci.config.from_file()
        # Initialize the translation client
        translation_client = init_translation_client(config)
        
        print("\nTesting Translation Service:")
        print("-" * 50)
        
        # Test the translation service  
        for text in TEST_TEXTS:
            print(f"\nOriginal text: {text}")
            
            # Test the translation service for each target language
            for target_lang in TARGET_LANGS:
                try:
                    # Create translation request details
                    translation_details = create_translation_details(text, target_lang)
                    # Translate the text
                    response = translation_client.batch_language_translation(
                        batch_language_translation_details=translation_details
                    )
                    # Get the translated text
                    translated_text = response.data.documents[0].translated_text
                    # Print the translated text
                    print(f"{target_lang.upper()}: {translated_text}")
                    
                except Exception as e:
                    # Print the error if the text is not translated
                    print(f"Error translating to {target_lang}: {str(e)}")
            
        return True
        
    except Exception as e:
        # Print the error if the translation service is not set up
        print(f"Error setting up translation service: {str(e)}")
        return False

# Initialize the language client
def init_language_client(config):
    """Initialize the language client."""
    return oci.ai_language.AIServiceLanguageClient(config)

# Create sentiment analysis request details
def create_sentiment_details(text):
    """Create sentiment analysis request details."""
    return oci.ai_language.models.BatchDetectLanguageSentimentsDetails(
        documents=[
            oci.ai_language.models.DominantLanguageDocument(
                key="1",
                text=text
            )
        ]
    )

# Test the sentiment analysis service   
def test_sentiment_analysis():
    """Test the sentiment analysis service."""
    if not verify_config():
        return False
        
    try:
        # Get the config
        config = oci.config.from_file()
        # Initialize the language client
        language_client = init_language_client(config)
        
        print("\nTesting Sentiment Analysis:")
        print("-" * 50)
        
        # Test the sentiment analysis service
        for text in SENTIMENT_TEXTS:
            try:
                # Create sentiment analysis request details
                sentiment_details = create_sentiment_details(text)
                # Detect the sentiment
                response = language_client.batch_detect_language_sentiments(
                    batch_detect_language_sentiments_details=sentiment_details
                )
                # Get the sentiment
                sentiment = response.data.documents[0].sentiment
                # Get the sentiment score
                score = response.data.documents[0].sentiment_score
                # Print the sentiment and score
                print(f"\nText: {text}")
                print(f"Sentiment: {sentiment}")
                print(f"Score: {score}")
                
            except Exception as e:
                # Print the error if the sentiment analysis service is not set up
                print(f"Error analyzing sentiment: {str(e)}")
        
        return True
        
    except Exception as e:
        # Print the error if the sentiment analysis service is not set up
        print(f"Error setting up sentiment analysis: {str(e)}")
        return False

# Main function
if __name__ == "__main__": 
    print("Testing OCI AI Services...\n") # Print the testing OCI AI Services
    
    # Test translation
    translation_success = test_translation()
    
    if translation_success:
        print("\nTranslation tests completed successfully!") # Print the translation tests completed successfully
    else:
        print("\nTranslation tests failed!") # Print the translation tests failed
        
    # Test sentiment analysis
    sentiment_success = test_sentiment_analysis()
    
    if sentiment_success:
        print("\nSentiment analysis tests completed successfully!") # Print the sentiment analysis tests completed successfully
    else:
        print("\nSentiment analysis tests failed!") # Print the sentiment analysis tests failed

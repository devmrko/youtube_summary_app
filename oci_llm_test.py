import oci
import os
import json
from datetime import datetime

# Static variables
CONFIG_PATH = os.path.expanduser(os.getenv("OCI_CONFIG_PATH"))
KEY_PATH = os.path.expanduser(os.getenv("OCI_KEY_PATH"))
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyajqi26fkxly6qje5ysvezzrypapl7ujdnqfjq6hzo2loq"  # meta.llama-3.3-70b-instruct
RESULTS_DIR = "results"
TEST_PROMPT = "why sky is blue?"

# Model parameters
MODEL_PARAMS = {
    "max_tokens": 600, # Maximum number of tokens in the response
    "temperature": 1, # Temperature for the response, range is 0-1
    "frequency_penalty": 0, # Penalty for frequent tokens, range is 0-1
    "presence_penalty": 0, # Penalty for tokens that are already in the response, range is 0-1
    "top_p": 0.75, # Top-p value for the response, range is 0-1
    "top_k": -1 # Top-k value for the response, range is 0-1
}

# Verify OCI config files
def verify_config():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(KEY_PATH):
        print(f"Error: OCI config files not found")
        return False
        
    return True

# Save results to a JSON file
def save_results(prompt, chat_response, filename=None):
    """Save results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{RESULTS_DIR}/llm_test_results_{timestamp}.json"
    
    os.makedirs(RESULTS_DIR, exist_ok=True) # Create results directory if it doesn't exist
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response_text": str(chat_response.data.chat_response.choices[0].message.content[0].text),
        "raw_response": str(vars(chat_response))
    }
    
    with open(filename, 'w', encoding='utf-8') as f: # Write results to a JSON file
        json.dump(result, f, ensure_ascii=False, indent=2) # Write results to a JSON file
    
    return filename

# Test the OCI chat service
def test_llm():
    try:
        config = oci.config.from_file()
        print("\nTesting OCI Chat Service:")
        print("-" * 50)
        
        try:
            # Initialize the OCI chat service
            generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config, 
                service_endpoint=ENDPOINT, 
                retry_strategy=oci.retry.NoneRetryStrategy(), 
                timeout=(10,240)
            )
            chat_detail = oci.generative_ai_inference.models.ChatDetails() # Initialize the chat details

            content = oci.generative_ai_inference.models.TextContent() # Initialize the text content
            content.text = TEST_PROMPT # Set the test prompt
            message = oci.generative_ai_inference.models.Message() # Initialize the message
            message.role = "USER" # Set the message role
            message.content = [content] # Set the message content
            chat_request = oci.generative_ai_inference.models.GenericChatRequest() # Initialize the chat request
            chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC # Set the API format
            chat_request.messages = [message] # Set the messages
            
            # Use MODEL_PARAMS
            for key, value in MODEL_PARAMS.items():
                setattr(chat_request, key, value) # Set the model parameters

            chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MODEL_ID) # Set the serving mode
            chat_detail.chat_request = chat_request # Set the chat request
            chat_detail.compartment_id = config["tenancy"] # Set the compartment ID
            chat_response = generative_ai_inference_client.chat(chat_detail) # Get the chat response
            
            print(vars(chat_response)) # Print the chat response
            print("**************************Chat Result**************************")
            print("\nRaw response:")
            print(chat_response.data) # Print the chat response data

            filename = save_results(TEST_PROMPT, chat_response) # Save the results to a JSON file
            print(f"\nResults saved to: {filename}") # Print the results saved to a JSON file
            
        except Exception as e:
            print(f"\nError during chat: {str(e)}") # Print the error during chat
            if hasattr(e, 'message'):
                print(f"Error message: {e.message}") # Print the error message
            if hasattr(e, 'code'):
                print(f"Error code: {e.code}") # Print the error code
            if hasattr(e, 'status'):
                print(f"Error status: {e.status}") # Print the error status
            
    except Exception as e:
        print(f"Error setting up service: {str(e)}") # Print the error setting up service
        return False

if __name__ == "__main__":
    print("Testing OCI Generative AI Service...\n") # Print the testing OCI Generative AI Service
    test_llm() # Test the OCI chat service

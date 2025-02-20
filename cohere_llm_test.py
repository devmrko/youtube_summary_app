import oci
import os
import json
from datetime import datetime

# Static variables
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
MODEL_ID = "cohere.command-r-plus-08-2024"
RESULTS_DIR = "results"
TEST_PROMPT = "what's LLM?"

# Model parameters
MODEL_PARAMS = {
    "max_tokens": 600,
    "temperature": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "top_p": 0.75,
    "top_k": 100
}

# Initialize the OCI client
def init_client(config):
    """Initialize the OCI client."""
    return oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config, 
        service_endpoint=ENDPOINT, 
        retry_strategy=oci.retry.NoneRetryStrategy(), 
        timeout=(10,240)
    )

# Create a chat request with the given prompt
def create_chat_request(prompt):
    """Create a chat request with the given prompt."""
    chat_request = oci.generative_ai_inference.models.CohereChatRequest()
    chat_request.message = prompt
    for key, value in MODEL_PARAMS.items():
        setattr(chat_request, key, value)
    return chat_request

# Create chat details with the given request
def create_chat_details(config, chat_request):
    """Create chat details with the given request."""
    chat_details = oci.generative_ai_inference.models.ChatDetails()
    chat_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        model_id=MODEL_ID
    )
    chat_details.chat_request = chat_request
    chat_details.compartment_id = config["tenancy"]
    return chat_details

# Save results to a JSON file
def save_results(prompt, chat_response, filename=None):
    """Save results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{RESULTS_DIR}/cohere_test_results_{timestamp}.json"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Print raw response for debugging
    print("\nRaw Response Data:")
    print(chat_response.data)
    print("\nResponse Data Attributes:")
    print(dir(chat_response.data))
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response_text": str(chat_response.data),
        "raw_response": str(vars(chat_response))
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return filename

# Test the Cohere chat service
def test_cohere():
    try:
        config = oci.config.from_file()
        print("\nTesting Cohere Chat Service:")
        print("-" * 50)
        
        try:
            # Initialize the OCI client
            client = init_client(config)
            # Create a chat request with the given prompt
            chat_request = create_chat_request(TEST_PROMPT)
            # Create chat details with the given request
            chat_details = create_chat_details(config, chat_request)
            # Get the chat response
            response = client.chat(chat_details)

            
            print("\n**************************Cohere Result**************************")
            print("\nFull Response:")
            print(response.data)
            
            # Save the results to a JSON file
            filename = save_results(TEST_PROMPT, response)
            print(f"\nResults saved to: {filename}")
            
        except Exception as e:
            # Print the error if the chat service is not set up
            print(f"\nError during chat: {str(e)}")
            # Print the error message
            if hasattr(e, 'message'):
                print(f"Error message: {e.message}")
            # Print the error code
            if hasattr(e, 'code'):
                print(f"Error code: {e.code}")
            # Print the error status
            if hasattr(e, 'status'):
                print(f"Error status: {e.status}")

    except Exception as e:
        # Print the error if the chat service is not set up
        print(f"Error setting up service: {str(e)}")
        return False

# Main function
if __name__ == "__main__":
    print("Testing Cohere Service...\n") # Print the testing Cohere Service
    test_cohere() 
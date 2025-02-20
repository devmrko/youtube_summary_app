import oci
import os
import json
from datetime import datetime

# Static variables
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
MODEL_ID = "meta.llama-3.3-70b-instruct"
RESULTS_DIR = "results"
TEST_PROMPT = "what's OCI?"

# Model parameters
MODEL_PARAMS = {
    "max_tokens": 600,
    "temperature": 1,
    "top_p": 0.75
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

# Create message content with the given prompt
def create_message_content(prompt):
    """Create message content with the given prompt."""
    content = oci.generative_ai_inference.models.TextContent()
    content.text = prompt
    content.type = "TEXT"
    return content

# Create message with the given content
def create_message(content):
    """Create message with the given content."""
    message = oci.generative_ai_inference.models.Message()
    message.role = "USER"
    message.content = [content]
    return message

# Create chat request with the given message
def create_chat_request(message):
    """Create chat request with the given message."""
    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.messages = [message]
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
        filename = f"{RESULTS_DIR}/llama_test_results_{timestamp}.json"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response_text": str(chat_response.data.chat_response.choices[0].message.content[0].text),
        "raw_response": str(vars(chat_response))
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return filename

# Test the Llama chat service
def test_llama():
    try:
        config = oci.config.from_file()
        print("\nTesting Llama Chat Service:")
        print("-" * 50)
        
        try:
            # Initialize the OCI client
            client = init_client(config)
            # Create message content with the given prompt
            content = create_message_content(TEST_PROMPT)
            # Create message with the given content
            message = create_message(content)
            # Create chat request with the given message
            chat_request = create_chat_request(message)
            # Create chat details with the given request
            chat_details = create_chat_details(config, chat_request)
            # Get the chat response
            response = client.chat(chat_details)
            # Print the generated text
            print("\n**************************Llama Result**************************")
            print(f"Generated Text: {response.data.chat_response.choices[0].message.content[0].text}")
            
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
    print("Testing Llama Service...\n") # Print the testing Llama Service
    test_llama() 
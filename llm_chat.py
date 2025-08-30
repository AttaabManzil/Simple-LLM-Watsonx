from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM
import gradio as gr

params = {
    GenParams.MAX_NEW_TOKENS: 256, 
    GenParams.TEMPERATURE: 0.5,
}

watsonx_llm = WatsonxLLM(
    model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
    url="https://us-south.ml.cloud.ibm.com",
    params=params,
    project_id = "skills-network"
)

def generate_response(query):
    response = watsonx_llm.invoke(query)
    return response


chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging='never',
    inputs = gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs = gr.Textbox(label="Output"),
    title = "Watson.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

chat_application.launch(server_name="127.0.0.1", server_port= 7860)

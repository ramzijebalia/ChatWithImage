import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from tools import ImageCaptionTool , ObjectDetectiionToool
import tempfile

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


tools = [ImageCaptionTool(), ObjectDetectiionToool()]


#we need the agent to remember his previous responses
Conversational_Memory = ConversationBufferMemory(
    memory_key='chat_history',
    k=5 , # the size of the agent memeory ( the agent remember back to 5 msg )
    return_messages=True
)

# give the agent the ability to communicate 
llm = ChatOpenAI(
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    temperature=0, # 0: the agent will be less creative , the agent will be more deterministic
    model_name= "gpt-3.5-turbo"
)


agent =  initialize_agent(
    agent="chat-conversational-react-description",
    tools = tools,
    llm=llm,
    max_iteration=5 , # tell the agent to trynot too hard ( the question will be too straight forward), just 5 iterations
    verbose= True,
    memory = Conversational_Memory,
    early_stoppy_method= 'generate'
)




# set tile
st.title("Ask a question to the image")

# set header
st.header("This app will help you ask questions about an image. Plesae upload an Image")

# upload file
file = st.file_uploader("" , type=["jpeg", "jpg" , "png"])
if file :
    # dipslay image
    st.image(file , use_column_width=True)

    # text input
    user_question = st.text_input("Ask a question about your image :" )
    
    # agnet response
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "temp_file")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.getbuffer())

        # Agent response 
        if user_question and user_question!="" :
            with st.spinner(text="In progress..."):
                response = agent.run('{} , this is the image path: {}'.format(user_question , temp_file_path))
                st.write(response)

import os
import toml
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import asyncio
import openai_async
import streamlit as st
from collections import deque

class BaseChatbot:
    def _load_api_key(self, provider: str) -> str:
        """Load the API key from a TOML file."""
        try:
            config = toml.load("SearchShellGPT.toml")
            return config[provider]['api_key']
        except Exception as e:
            raise RuntimeError(f"Failed to load API key from config.toml for {provider}") from e

    def search_web(self, query: str, num_results: int = 3) -> list[dict]:
        """Search the web only when needed."""
        try:
            results = []
            with st.spinner('🔍 Searching the web...'):
                for url in search(query, num_results=num_results):
                    results.append({
                        'href': url,
                        'title': self._get_page_title(url),
                        'body': self.extract_content(url)
                    })
            return results
        except Exception as e:
            st.error(f"Error searching web: {e}")
            return []

    def _get_page_title(self, url: str) -> str:
        """Extract the title from a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.title.string.strip() if soup.title else url
        except Exception:
            return url

    def extract_content(self, url: str) -> str:
        """Extract main content from a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()

            main_content = (soup.find('main') or soup.find('article') or 
                          soup.find('div', {'class': ['content', 'main', 'article']}))
            
            text = main_content.get_text(separator='\n', strip=True) if main_content else \
                   soup.get_text(separator='\n', strip=True)
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines)[:2000]
        except Exception as e:
            st.error(f"Error extracting content from {url}: {e}")
            return ""

class OpenAIChatbot(BaseChatbot):
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.api_key = self._load_api_key("openai")

    async def chat(self, user_input: str, chat_history: list, should_search: bool = False) -> str:
        try:
            messages = self._build_chat_history(chat_history)
            
            if should_search or any(trigger in user_input.lower() 
                                  for trigger in ['search', 'look up', 'find out', 'what is', 'who is']):
                web_results = self.search_web(user_input)
                context = "\n\n".join(f"{r['title']}\n{r['body']}" for r in web_results)
                
                if context.strip():
                    messages.append({
                        "role": "system",
                        "content": f"Additional context from web search:\n\n{context}"
                    })

            messages.append({"role": "user", "content": user_input})

            response = await openai_async.chat_complete(
                api_key=self.api_key,
                timeout=30,
                payload={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            )

            return response.json()['choices'][0]['message']['content'].strip()

        except Exception as e:
            return f"Error: {str(e)}"

    def _build_chat_history(self, chat_history: list) -> list:
        """Build the chat history in the format required by the API."""
        messages = [{
            "role": "system",
            "content": "You are a helpful AI assistant with access to web search capabilities. "
                      "You can search the internet when needed to provide up-to-date information. "
                      "Always maintain context of the conversation and provide accurate, relevant responses."
                      "ALWAYS incorporate emojis wherever possible and relevant to make your answers interesting.\n\n"
        }]
        
        for role, content in chat_history:
            messages.append({"role": role, "content": content})
            
        return messages

import aiohttp

class GeminiChatbot(BaseChatbot):
    def __init__(self, model_name: str = "gemini-1.5-flash-8b"):
        self.model_name = model_name
        self.api_key = self._load_api_key("gemini")
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    async def chat(self, user_input: str, chat_history: list, should_search: bool = False) -> str:
        try:
            conversation = self._build_chat_history(chat_history)

            if should_search or any(trigger in user_input.lower() 
                                  for trigger in ['search', 'look up', 'find out', 'what is', 'who is']):
                web_results = self.search_web(user_input)
                context = "\n\n".join(f"{r['title']}\n{r['body']}" for r in web_results)
                if context.strip():
                    conversation += f"\nAdditional context from web search:\n\n{context}"

            conversation += f"\nUser: {user_input}\nAssistant: "

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }

            data = {
                "contents": [{
                    "parts": [{
                        "text": conversation
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}?key={self.api_key}",
                    headers=headers,
                    json=data,
                    timeout=30
                ) as response:

                    response.raise_for_status()
                    response_data = await response.json()

                    try:
                        return response_data['candidates'][0]['content']['parts'][0]['text']
                    except (KeyError, IndexError) as e:
                        return f"Error parsing Gemini response: {str(e)}"

        except Exception as e:
            return f"Error: {str(e)}"


    def _build_chat_history(self, chat_history: list) -> str:
        """Build the chat history in the format required by Gemini API."""
        conversation = "You are a helpful AI assistant with access to web search capabilities. " \
                      "You can search the internet when needed to provide up-to-date information. " \
                      "Always maintain context of the conversation and provide accurate, relevant responses. " \
                      "ALWAYS incorporate emojis wherever possible and relevant to make your answers interesting.\n\n"
        
        for role, content in chat_history:
            prefix = "User: " if role == "user" else "Assistant: "
            conversation += f"{prefix}{content}\n\n"
            
        return conversation


def create_message_container():
    """Create a container for chat messages with scrolling."""
    return st.container()

def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""  # Initialize user_input

def display_messages(container):
    """Display chat messages in the container."""
    for message in st.session_state.messages:
        role, content = message
        with container.container():
            if role == "user":
                st.markdown(f'<div style="text-align: right; margin-bottom: 10px;"><span style="background-color: #3b3232; padding: 8px 12px; border-radius: 15px; display: inline-block; max-width: 70%;"><b>You:</b> {content}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="margin-bottom: 10px;"><span style="background-color: #103e41; padding: 8px 12px; border-radius: 15px; display: inline-block; max-width: 70%;"><b>🤖 Assistant:</b> {content}</span></div>', unsafe_allow_html=True)

async def async_chat(chatbot, user_input, chat_history, should_search):
    response = await chatbot.chat(user_input, chat_history, should_search)
    model_name = "GPT" if isinstance(chatbot, OpenAIChatbot) else "Gemini"
    return response, model_name

def main():
    st.set_page_config(page_title="AI Chatbot with Web Search", page_icon="🤖", layout="wide")
    
    # Initialize session state
    init_session_state()
    
    # Custom CSS for layout
    st.markdown("""
        <style>
        .stApp {
            max-width: 1920px;
            margin: 0 auto;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 150px); /* Adjusted to account for any headers */
        }
        .chat-messages {
            flex-grow: 1;
            overflow-y: auto;
            padding: 20px;
            margin-bottom: 60px; /* Adjusted for better spacing */
        }
        .chat-input {
            position: fixed; /* Use sticky instead of fixed for better behavior */
            bottom: 0;
            background-color: white;
            padding: 20px;
            z-index: 1000;
        }
        .stButton button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)


    # Sidebar
    with st.sidebar:
        st.title("Chat Settings")
        should_search = st.checkbox("Enable Web Search", value=True)
        model_type = st.radio("Select Model", ("GPT", "Gemini"))
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Update chatbot based on selected model
    if model_type == "GPT":
        if not isinstance(st.session_state.chatbot, OpenAIChatbot):
            st.session_state.chatbot = OpenAIChatbot()
    else:
        if not isinstance(st.session_state.chatbot, GeminiChatbot):
            st.session_state.chatbot = GeminiChatbot()

    # Main chat interface
    st.title("🤖 AI Chatbot with Web Search")
    
    # Create message container
    message_container = create_message_container()
    
    # Display messages
    display_messages(message_container)
    
    # Chat input at the bottom
    with st.container():
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        cols = st.columns([8, 1])
        with cols[0]:
            user_input = st.text_input("Message", key="user_input", value="", label_visibility="collapsed")
        with cols[1]:
            send_button = st.button("Send", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Handle chat interaction
    if send_button and user_input.strip():
        # Add user message to state
        st.session_state.messages.append(("user", user_input))
        
        # Get chatbot response
        with st.spinner('Thinking...'):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response, model_name = loop.run_until_complete(async_chat(
                st.session_state.chatbot, 
                user_input, 
                st.session_state.messages, 
                should_search
            ))
            loop.close()
        
        # Add assistant response to state with model name
        st.session_state.messages.append(("assistant", f"[Model: {model_name}] {response}"))
        
        # Clear input and rerun to update UI
        #st.session_state.user_input = ""  # Reset input box
        st.rerun()


if __name__ == "__main__":
    main()

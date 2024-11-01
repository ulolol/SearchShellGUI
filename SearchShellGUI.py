import os
import toml
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import asyncio
import openai_async
import streamlit as st
from collections import deque

class OpenAIWebWrapper:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.api_key = self._load_api_key("openai")

    def _load_api_key(self, provider: str) -> str:
        """Load the API key from a TOML file."""
        try:
            config = toml.load("SearchShellGPT.toml")
            return config[provider]['api_key']
        except Exception as e:
            raise RuntimeError(f"Failed to load API key from config.toml for {provider}") from e

    def search_web(self, query: str, num_results: int = 5) -> list[dict]:
        """Search the web using Google and return results."""
        try:
            results = []
            st.write(f"Searching for: {query}")
            for url in search(query, num_results=num_results):
                st.write(f"Found URL: {url}")
                results.append({
                    'href': url,
                    'title': self._get_page_title(url),
                    'body': self.extract_content(url)
                })
            if not results:
                st.error("No results were found.")
            return results
        except Exception as e:
            st.error(f"Error searching web: {e}")
            return []


    def _get_page_title(self, url: str) -> str:
        """Extract the title from a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else url
            return title.strip()
        except Exception:
            return url

    def extract_content(self, url: str) -> str:
        """Extract main content from a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript']):
                element.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'main', 'article']})
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)

            return cleaned_text[:2000]
        except Exception as e:
            st.error(f"Error extracting content from {url}: {e}")
            return ""

    async def query_openai_async(self, prompt: str, context: str) -> str:
        """Query OpenAI with the given prompt and context using openai-async."""
        try:
            if not context.strip():
                return "No context was found from web searches. The model will provide a general response without current information."

            full_prompt = (
                f"Context from web searches:\n\n{context}\n\n"
                f"Question: {prompt}\n\n"
                f"Please provide a comprehensive answer based on the context above. "
                f"Please ensure the answer is detailed with points wherever necessary. "
                f"Please ensure that the answer is properly formatted for reading. "
                f"If the context doesn't contain relevant information, please state that clearly."
            )

            response = await openai_async.chat_complete(
                api_key=self.api_key,
                timeout=30,
                payload={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": full_prompt}
                    ],
                    "max_tokens": 2000
                }
            )

            response_json = response.json()
            return response_json['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error querying OpenAI: {e}"

class GeminiWebWrapper(OpenAIWebWrapper):
    def __init__(self, model_name: str = "gemini-1.5-flash-8b"):
        super().__init__(model_name)
        self.api_key = self._load_api_key("gemini")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"

    async def query_gemini_async(self, prompt: str, context: str) -> str:
        """Query Gemini with the given prompt and context using direct API call."""
        try:
            if not context.strip():
                return "No context was found from web searches. The model will provide a general response without current information."

            full_prompt = (
                f"Context from web searches:\n\n{context}\n\n"
                f"Question: {prompt}\n\n"
                f"Please provide a comprehensive answer based on the context above. "
                f"Please ensure the answer is detailed with points wherever necessary. "
                f"ALWAYS incorporate emojis wherever possible and relevant to make your answers interesting"
                f"Please ensure that the answer is properly formatted for reading. "
                f"If the context doesn't contain relevant information, please state so clearly, and instead \
                    provide whatever info you have on the topic."
            )

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }

            data = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }

            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )

            response.raise_for_status()
            response_data = response.json()

            # Extract the generated text from the response
            try:
                return response_data['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError) as e:
                return f"Error parsing Gemini response: {str(e)}"

        except requests.exceptions.RequestException as e:
            return f"Error querying Gemini API: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"


def main():
    # Set the page configuration with title and icon
    st.set_page_config(page_title="SearchShell", page_icon="🔍")

    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_type = st.sidebar.radio("Select Model", ("GPT", "Gemini"))

    # Slider for selecting number of search results
    num_results = st.sidebar.slider("Number of Search Results", min_value=1, max_value=10, value=3)

    # Main chatbot area with page title
    st.title("SearchShell")
    chat_history = deque(maxlen=10)

    # Create a form for user input
    with st.form(key='chat_form'):
        user_input = st.text_input("You:", value="", key="user_input")
        submit_button = st.form_submit_button(label='Send')

    # Check if the form is submitted
    if submit_button and user_input.strip():
        #st.write(f"User input received: {user_input}")
        chat_history.append(("User", user_input))

        if model_type == "GPT":
            wrapper = OpenAIWebWrapper()
        else:
            wrapper = GeminiWebWrapper()

        # Search the web with selected number of results
        web_results = wrapper.search_web(user_input, num_results)
        if not web_results:
            st.error("No web search results found.")

        # Build context from web results
        context = "\n\n".join(f"{result['title']}\n{result['body']}" for result in web_results)
        if not context.strip():
            st.info("No context was found from web searches. The model will provide a general response without current information.")
        else:
            st.write("Context obtained from web searches. Ingesting ... ")
            #st.write(context)

        # Async query to the selected model
        if model_type == "GPT":
            response = asyncio.run(wrapper.query_openai_async(user_input, context))
        else:
            response = asyncio.run(wrapper.query_gemini_async(user_input, context))

        chat_history.append(("Bot", response))

    # Display chat history
    for sender, message in chat_history:
        if sender == "User":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Chatbot:** {message}")

if __name__ == "__main__":
    main()

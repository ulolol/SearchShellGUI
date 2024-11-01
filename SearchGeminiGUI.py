import requests
from bs4 import BeautifulSoup
from googlesearch import search
import time
import json
import toml
from typing import List, Dict
import streamlit as st

class GeminiWebWrapper:
    def __init__(self):
        self.api_key = self.load_api_key()
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"

    def load_api_key(self) -> str:
        """Load the Gemini API key from a TOML file."""
        try:
            config = toml.load("SearchShellGPT.toml")
            return config['gemini']['api_key']
        except Exception as e:
            raise RuntimeError("Failed to load API key from SearchShellGPT.toml") from e

    def search_web(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search the web using Google and return results."""
        try:
            results = []
            for url in search(query, num_results=num_results):
                results.append({
                    'href': url,
                    'title': self._get_page_title(url),
                    'body': ''
                })
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

    def generate_context(self, query: str, num_results: int = 3) -> str:
        """Generate context from web search results."""
        results = self.search_web(query, num_results)
        context = []

        for result in results:
            url = result['href']
            title = result['title']

            st.info(f"Fetching content from: {url}")
            content = self.extract_content(url)

            if content:
                context_entry = (
                    f"Source: {title}\n"
                    f"URL: {url}\n\n"
                    f"Content:\n{content}\n"
                    f"{'='*50}\n"
                )
                context.append(context_entry)
            else:
                context_entry = (
                    f"Source: {title}\n"
                    f"URL: {url}\n"
                    f"{'='*50}\n"
                )
                context.append(context_entry)

            time.sleep(1)

        return "\n".join(context)

    def query_gemini(self, prompt: str, context: str) -> str:
        """Query Gemini with the given prompt and context using direct API call."""
        try:
            if not context.strip():
                return "No context was found from web searches. The model will provide a general response without current information."

            full_prompt = (
                f"Context from web searches:\n\n{context}\n\n"
                f"Question: {prompt}\n\n"
                f"Please provide a comprehensive answer based on the context above. "
                f"Please ensure the answer is detailed with points wherever necessary. "
                f"Use emojis wherever needed to make your answers interesting"
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
    st.title("Search Shell")
    st.write("Type a search query and get a response from the Gemini 1.5 Flash model.")

    query = st.text_input("Search Query", placeholder="Enter your search query")
    show_context = st.checkbox("Show Context")
    num_results = st.number_input("Number of Results", min_value=1, max_value=10, value=3, step=1)

    if st.button("Search"):
        wrapper = GeminiWebWrapper()
        context = wrapper.generate_context(query, num_results)

        if show_context:
            st.write("Context gathered from web:")
            st.write(context)
            st.write("Generating response...")

        response = wrapper.query_gemini(query, context)
        st.write("Response:")
        st.write(response)

if __name__ == "__main__":
    main()
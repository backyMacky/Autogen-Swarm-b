import os
import random
import string
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from flask import Flask, request, render_template_string, flash
import autogen
import logging
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import yake
import bleach  # Added for sanitization
import time

# Import for File Search
from typing_extensions import override

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Create logs directory if it doesn't exist
LOGS_DIR = 'logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Function to generate log filename
def generate_log_filename():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    filename = f"log_{timestamp}_{random_str}.txt"
    return os.path.join(LOGS_DIR, filename)

# -----------------------------------
# Configuration Settings
# -----------------------------------

@dataclass
class Config:
    """Configuration settings"""
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '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')
    MODEL: str = "gpt-4o"
    TEMPERATURE: float = 1
    MAX_ROUND: int = 10
    SECRET_KEY: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'your-nla5xu5WdvWSqEjd8wsPnTlJtB5iXZTY1cuhcRQo798AGov9XycghxB8t5hB82SNlxGSqSDUTET3BlbkFJLIxGq8wpC2K-key'))
    ASSISTANT_ID: str = field(default_factory=lambda: os.getenv('ASSISTANT_ID', 'asst_a2rbz1zXNB691acwLF9vsCng'))
    VECTOR_STORE_IDS: List[str] = field(default_factory=lambda: os.getenv('VECTOR_STORE_IDS', 'vs_ECcEhMjKPr3sMw7EOD3MfnaZ').split(','))  # Comma-separated IDs
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', 'AIzaSyBkNjKsyVnhsOjwkMM4KT0juEwK9ZeH1ao')
    GOOGLE_SEARCH_ENGINE_ID: str = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '134c38d750ae84760')

    # WebScraper specific settings
    SCRAPER_TRUSTED_DOMAINS: set = field(default_factory=lambda: {
        'wikipedia.org', 'forbes.com', 'mckinsey.com', 'wsj.com',
        'ft.com', 'reuters.com', 'bloomberg.com', 'hbr.org',
        'techcrunch.com', 'inc.com'
    })
    SCRAPER_BLOCKED_DOMAINS: set = field(default_factory=lambda: set())
    SCRAPER_USER_AGENTS: list = field(default_factory=lambda: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
    ])
    SCRAPER_MIN_CONTENT_LENGTH: int = 350
    SCRAPER_MAX_SPECIAL_CHAR_RATIO: float = 0.3
    SCRAPER_MIN_AVG_WORD_LENGTH: float = 2.0
    SCRAPER_MAX_AVG_WORD_LENGTH: float = 15.0
    SCRAPER_EXCERPT_LENGTH: int = 1000
    SCRAPER_RATE_LIMIT_DELAY: int = 2

    @property
    def llm_config(self) -> Dict:
        return {
            "config_list": [{
                "model": self.MODEL,
                "api_key": self.OPENAI_API_KEY
            }],
            "temperature": self.TEMPERATURE
        }

# -----------------------------------
# Web Scraping and Keyword Analysis
# -----------------------------------

import os
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
import json
import time
import logging

class WebScraper:
    """Web scraping with focus on context extraction from both trusted and general sources"""
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.request_timestamps = []
        
        # Create logs directory
        self.logs_dir = 'scraper_logs'
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        # Initialize logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.logs_dir, f'scraper_log_{timestamp}.txt')
        self.cache_file = os.path.join(self.logs_dir, f'cache_info_{timestamp}.json')
        
        # Set up file handler for scraper-specific logging
        self.logger = logging.getLogger('webscraper')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def _get_headers(self):
        return {
            'User-Agent': random.choice(self.config.SCRAPER_USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1'
        }

    def _is_valid_content(self, text: str) -> bool:
        """Basic content quality checks"""
        if not text or not text.strip():
            return False
        
        if len(text) < self.config.SCRAPER_MIN_CONTENT_LENGTH:
            return False
            
        special_char_ratio = len([c for c in text if not c.isalnum() and not c.isspace()]) / len(text)
        if special_char_ratio > self.config.SCRAPER_MAX_SPECIAL_CHAR_RATIO:
            return False
            
        words = text.split()
        if not words:
            return False
            
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < self.config.SCRAPER_MIN_AVG_WORD_LENGTH or avg_word_length > self.config.SCRAPER_MAX_AVG_WORD_LENGTH:
            return False
            
        return True

    def _clean_text(self, text: str) -> str:
        """Clean and format text for compact logging"""
        if not text:
            return ""
        text = ' '.join(text.split())
        text = ' '.join(line.strip() for line in text.splitlines() if line.strip())
        text = ' '.join(s.strip('.,;') for s in text.split())
        return text.strip()

    def _extract_relevant_text(self, soup: BeautifulSoup, query: str) -> str:
        """Extract most relevant text snippets based on query keywords"""
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            element.decompose()

        keywords = set(query.lower().split())
        relevant_paragraphs = []
        
        for p in soup.find_all(['p', 'article', 'section', 'div']):
            text = self._clean_text(p.get_text())
            if text and len(text) > self.config.SCRAPER_MIN_CONTENT_LENGTH:
                text_lower = text.lower()
                keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
                if keyword_matches > 0 and self._is_valid_content(text):
                    relevant_paragraphs.append((text, keyword_matches))
        
        if not relevant_paragraphs:
            return ""
            
        relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
        selected_texts = [p[0] for p in relevant_paragraphs[:2]]
        combined_text = ' '.join(selected_texts)

        if len(combined_text) > self.config.SCRAPER_EXCERPT_LENGTH:
            truncated = combined_text[:self.config.SCRAPER_EXCERPT_LENGTH]
            last_period = truncated.rfind('.')
            if last_period > 0:
                combined_text = truncated[:last_period + 1]
            else:
                combined_text = truncated + '...'
        
        return combined_text

    def google_search(self, query: str, limit: int = 3) -> list:
        """Perform Google Custom Search including all domains"""
        try:
            search_url = (
                f"https://www.googleapis.com/customsearch/v1"
                f"?key={self.config.GOOGLE_API_KEY}"
                f"&cx={self.config.GOOGLE_SEARCH_ENGINE_ID}"
                f"&q={requests.utils.quote(query)}"
                f"&num={limit * 2}"
            )
            
            response = requests.get(search_url, headers=self._get_headers())
            response.raise_for_status()
            
            results = response.json()
            urls = []
            
            for item in results.get('items', []):
                url = item['link']
                domain = urlparse(url).netloc
                
                if domain in self.config.SCRAPER_BLOCKED_DOMAINS:
                    continue
                
                urls.append({
                    'url': url,
                    'domain': domain,
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'trusted': domain in self.config.SCRAPER_TRUSTED_DOMAINS
                })
            
            urls.sort(key=lambda x: (not x['trusted']))
            
            valid_urls = len([u for u in urls if u['domain'] and u['snippet']])
            if valid_urls > 0:
                self.logger.info(f"Found {valid_urls} valid search results for: {query}")
            
            return urls[:limit]
            
        except Exception as e:
            self.logger.error(f"Google search error: {str(e)}")
            return []

    def scrape_url(self, url: str, query: str = None) -> str:
        """Scrape and extract relevant context from a URL"""
        try:
            cache_key = f"{url}_{query if query else ''}"
            if cache_key in self.cache:
                timestamp, content = self.cache[cache_key]
                if (datetime.now() - datetime.fromtimestamp(timestamp)).total_seconds() < 3600:
                    return content

            time.sleep(self.config.SCRAPER_RATE_LIMIT_DELAY)
            
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            domain = urlparse(url).netloc
            
            content = self._extract_relevant_text(soup, query) if query else ""
            if not content:
                title = self._clean_text(soup.title.string) if soup.title else ""
                content = ""
                for p in soup.find_all('p'):
                    text = self._clean_text(p.get_text())
                    if text and len(content) < self.config.SCRAPER_EXCERPT_LENGTH:
                        content += " " + text
                    elif len(content) >= self.config.SCRAPER_EXCERPT_LENGTH:
                        break
                content = content.strip()
                if title and content:
                    content = f"{title}: {content}"
                content = content[:self.config.SCRAPER_EXCERPT_LENGTH] + "..." if content else ""

            if not content:
                return ""

            if not self._is_valid_content(content):
                return ""
                
            trust_status = "Trusted" if domain in self.config.SCRAPER_TRUSTED_DOMAINS else "General"
            formatted_content = f"Source ({trust_status}): {domain} | {content}"
            
            # Only log if we have valid content
            self.logger.info(formatted_content)
            self.cache[cache_key] = (time.time(), formatted_content)
            return formatted_content
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return ""

    def gather_articles(self, query: str, limit: int = 5) -> list:
        """Gather articles from both trusted and general sources while maintaining quality"""
        self.logger.info(f"Starting article gathering for query: '{query}'")
        articles = []
        
        search_results = self.google_search(query, limit=limit)
        trusted_count = general_count = 0
        
        for result in search_results:
            content = self.scrape_url(result['url'], query)
            if content:  # Only process if we got valid content
                if result['trusted']:
                    trusted_count += 1
                else:
                    general_count += 1
                articles.append(content)
            
            if len(articles) >= limit:
                break
        
        if articles:  # Only log summary if we found any articles
            self.logger.info(f"Collection complete: {trusted_count} trusted and {general_count} general articles")
        
        return articles

    def _save_cache_info(self):
        """Save cache information to JSON file"""
        cache_info = {
            'timestamp': datetime.now().isoformat(),
            'total_cached_urls': len(self.cache),
            'cached_domains': list(set(urlparse(url.split('_')[0]) for url in self.cache.keys())),
            'cache_entries': [{
                'url': url.split('_')[0],
                'timestamp': data[0],
                'content_length': len(data[1])
            } for url, data in self.cache.items()]
        }
        
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_info, f, indent=4)

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            'total_cached': len(self.cache),
            'domains_cached': len(set(urlparse(url.split('_')[0]).netloc for url in self.cache.keys())),
            'cache_size_mb': sum(len(v[1]) for v in self.cache.values()) / (1024 * 1024),
            'timestamp': datetime.now().isoformat()
        }
        return stats

class KeywordAnalyzer:
    """Handles keyword extraction and analysis optimized for project brief analysis."""

    def __init__(self):
        # Business and project-relevant patterns
        self.business_patterns = {
            'industry': [
                'market', 'industry', 'sector', 'business', 'enterprise',
                'vertical', 'niche', 'segment', 'domain', 'field',
                'commercial', 'corporate', 'trade', 'startup', 'company'
            ],
            'technology': [
                'software', 'platform', 'system', 'application', 'technology',
                'api', 'cloud', 'database', 'interface', 'infrastructure',
                'framework', 'architecture', 'integration', 'automation', 'deployment',
                'backend', 'frontend', 'fullstack', 'mobile', 'web',
                'ai', 'ml', 'algorithm', 'analytics', 'data',
                'saas', 'paas', 'iaas', 'microservices', 'serverless'
            ],
            'metrics': [
                'revenue', 'growth', 'profit', 'sales', 'performance',
                'roi', 'margin', 'conversion', 'retention', 'churn',
                'arpu', 'cac', 'ltv', 'mrr', 'arr',
                'burn rate', 'runway', 'valuation', 'equity', 'funding'
            ],
            'operations': [
                'operations', 'process', 'workflow', 'logistics', 'supply chain',
                'inventory', 'procurement', 'distribution', 'fulfillment', 'delivery',
                'quality', 'compliance', 'standards', 'certification', 'audit'
            ],
            'product': [
                'product', 'service', 'solution', 'offering', 'portfolio',
                'feature', 'functionality', 'capability', 'specification', 'requirement',
                'usability', 'ux', 'ui', 'design', 'experience'
            ]
        }

        # Word relationships for better context understanding
        self.word_relationships = {
            'equivalents': {
                'app': ['application', 'software', 'platform'],
                'ai': ['artificial intelligence', 'machine learning', 'deep learning'],
                'ui': ['user interface', 'frontend', 'design'],
                'ux': ['user experience', 'usability', 'interface'],
                'api': ['interface', 'integration', 'endpoint'],
                'crm': ['customer relationship management', 'sales', 'customer']
            }
        }

        # Project-specific stop words
        self.project_stop_words = {
            'need', 'want', 'look', 'looking', 'searching',
            'find', 'help', 'please', 'would', 'like',
            'current', 'currently', 'existing', 'new',
            'development', 'develop', 'create', 'implementation',
            'get', 'getting', 'make', 'making', 'do',
            'doing', 'use', 'using', 'utilize', 'utilizing'
        }
        
        # Context score multipliers
        self.context_multipliers = {
            'industry': 1.5,
            'technology': 1.4,
            'metrics': 1.3,
            'operations': 1.2,
            'product': 1.2
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text without using regex."""
        if not text:
            return ""
        
        # Convert to lowercase and split
        text = text.lower()
        
        # Replace common punctuation with spaces
        for char in '.,!?()[]{};:"\'/\\':
            text = text.replace(char, ' ')
            
        # Normalize spaces
        words = text.split()
        text = ' '.join(words)
        
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations using word relationships."""
        for abbr, expansions in self.word_relationships['equivalents'].items():
            if abbr in text.lower():
                text = text + ' ' + ' '.join(expansions)
        return text

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extracts key phrases from project brief, optimized for web search.
        
        Args:
            text (str): The project brief to analyze.
            
        Returns:
            List[str]: A list of search-optimized keywords.
        """
        if not text:
            return []

        # Text preprocessing
        text = self._clean_text(text)
        text = self._expand_abbreviations(text)
        
        # Get stop words
        stop_words = set(stopwords.words('english')) | self.project_stop_words
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Extract candidate terms (single words and phrases)
        candidates = []
        
        # Single words
        single_words = [
            word for word in tokens 
            if word.isalnum() and 
            word not in stop_words and 
            len(word) > 2
        ]
        candidates.extend(single_words)
        
        # Generate phrases (2-3 words)
        for i in range(len(tokens) - 1):
            if all(t.isalnum() for t in tokens[i:i+2]):
                phrase = ' '.join(tokens[i:i+2])
                if not any(word in stop_words for word in phrase.split()):
                    candidates.append(phrase)
                    
        if len(tokens) > 2:
            for i in range(len(tokens) - 2):
                if all(t.isalnum() for t in tokens[i:i+3]):
                    phrase = ' '.join(tokens[i:i+3])
                    if not any(word in stop_words for word in phrase.split()):
                        candidates.append(phrase)

        # Score candidates
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_term_score(candidate)
            scored_candidates.append((candidate, score))

        # Sort and remove duplicates while maintaining original output format
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique_terms = []
        
        for term, score in scored_candidates:
            normalized_term = term.lower()
            if normalized_term not in seen and not any(normalized_term in s for s in seen):
                seen.add(normalized_term)
                unique_terms.append(term)

        return unique_terms[:8]

    def _calculate_term_score(self, term: str) -> float:
        """Calculate relevance score for a term."""
        score = 1.0
        term_lower = term.lower()
        
        # Pattern matching score
        for pattern_type, patterns in self.business_patterns.items():
            if any(pattern in term_lower for pattern in patterns):
                score *= self.context_multipliers.get(pattern_type, 1.0)
        
        # Length and phrase bonuses
        words = term.split()
        if len(words) > 1:
            score *= (1.0 + (len(words) * 0.2))  # Boost multi-word phrases
            
        # Terminology bonus
        for abbr, expansions in self.word_relationships['equivalents'].items():
            if abbr in term_lower or any(exp in term_lower for exp in expansions):
                score *= 1.2
                
        return score

    def analyze_intent(self, text: str) -> Dict:
        """
        Analyzes the project brief to identify key themes and statistics.
        Maintains exact original return structure for compatibility.
        
        Args:
            text (str): The project brief to analyze.
            
        Returns:
            Dict: Original format with top themes, keyword density, word count, 
                 and unique word statistics.
        """
        # Text preprocessing
        text = self._clean_text(text)
        
        # Tokenization and filtering
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
        
        # Word frequency calculation
        word_freq = Counter(filtered_tokens)
        
        # Get top themes (maintaining original format)
        themes = word_freq.most_common(5)
        
        # Calculate statistics (maintaining original format)
        unique_word_count = len(set(filtered_tokens))
        keyword_density = {word: freq / len(tokens) for word, freq in themes}
        
        # Return in original format
        return {
            "top_themes": themes,
            "keyword_density": keyword_density,
            "word_count": len(tokens),
            "unique_words": unique_word_count
        }

# -----------------------------------
# OpenAI Assistant Integration
# -----------------------------------

class OpenAIAssistant:
    """Handles interaction with OpenAI Assistant"""

    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.assistant_id = config.ASSISTANT_ID

    def get_expert_advice(self, query: str) -> str:
        """Get advice from the OpenAI Assistant"""
        try:
            # Create a thread
            thread = self.client.beta.threads.create()

            # Add message to the thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )

            # Create and run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )

            # Wait for the run to complete
            while run.status in ["queued", "in_progress"]:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

            # Retrieve the assistant's response
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            for msg in messages.data:
                if msg.role == 'assistant':
                    return msg.content

            return "No response from assistant."

        except Exception as e:
            logging.error(f"Error getting expert advice: {str(e)}")
            return f"Error consulting expert: {str(e)}"

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for the given text using OpenAI's embedding API."""
        try:
            response = self.client.Embedding.create(
                input=text,
                model="text-embedding-3-large"  # Example model
            )
            embeddings = response['data'][0]['embedding']
            return embeddings
        except Exception as e:
            logging.error(f"Error getting embeddings: {str(e)}")
            return []

# -----------------------------------
# Librarian Agent
# -----------------------------------

class LibrarianAgent(autogen.AssistantAgent):
    def __init__(
        self, 
        name: str, 
        llm_config: Dict, 
        system_message: str, 
        openai_assistant: 'OpenAIAssistant', 
        assistant_id: str, 
        vector_store_ids: List[str]
    ):
        super().__init__(name=name, llm_config=llm_config, system_message=system_message)
        self.openai_assistant = openai_assistant
        self.assistant_id = assistant_id
        self.vector_store_ids = vector_store_ids

        # Initialize OpenAI client
        self.client = OpenAI(api_key=llm_config["config_list"][0]["api_key"])

        # Enable file_search tool and attach vector stores
        self.enable_file_search()

    def enable_file_search(self):
        """Enable the file_search tool and attach vector stores to the assistant."""
        try:
            self.client.beta.assistants.update(
                assistant_id=self.assistant_id,
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": self.vector_store_ids}}
            )
            logging.info("File Search tool enabled and vector stores attached to Librarian Agent.")
        except Exception as e:
            logging.error(f"Failed to enable file_search tool: {str(e)}")

    def process_message(self, message: Dict) -> Dict:
        """
        Process incoming messages and respond using OpenAIAssistant with file_search context.

        Args:
            message (Dict): The incoming message containing at least the 'content' key.

        Returns:
            Dict: A dictionary representing the assistant's response.
        """
        content = message.get('content', '').strip()
        if not content:
            return {'role': 'assistant', 'content': 'No content to process.'}

        try:
            # Step 1: Create a new thread with the user message
            thread = self.client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            logging.info(f"Thread created with ID: {thread.id}")

        except Exception as e:
            logging.error(f"Error creating thread: {str(e)}")
            return {'role': 'assistant', 'content': f"Error creating thread: {str(e)}"}

        try:
            # Step 2: Create and poll the run until it's in a terminal state
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            logging.info(f"Run created with ID: {run.id} and status: {run.status}")

            # Step 3: Retrieve messages from the thread and run
            messages = list(self.client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
            logging.info(f"Retrieved {len(messages)} messages from run ID: {run.id}")

            if not messages:
                logging.warning("No messages retrieved from the run.")
                return {'role': 'assistant', 'content': "I'm sorry, I couldn't retrieve any information."}

            # Step 4: Process the assistant's response and annotations
            # Assuming the assistant's message is the first one with role 'assistant'
            assistant_messages = [msg for msg in messages if msg.role == 'assistant']
            if not assistant_messages:
                logging.warning("No assistant messages found in the run.")
                return {'role': 'assistant', 'content': "I'm sorry, I couldn't retrieve any information."}

            assistant_message = assistant_messages[0]
            message_content = assistant_message.content[0].text  # Assuming content is a list with 'text'

            # Extract annotations if available
            annotations = getattr(message_content, 'annotations', [])

            citations = []
            for index, annotation in enumerate(annotations):
                # Replace annotation text with citation index
                message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
                # Retrieve the cited file information
                file_citation = getattr(annotation, "file_citation", None)
                if file_citation:
                    cited_file = self.client.files.retrieve(file_citation.file_id)
                    citations.append(f"[{index}] {cited_file.filename}")

            # Prepare the assistant's response
            assistant_response = message_content.value.strip()
            if not assistant_response:
                # Fallback to expert advice if no response is found
                expert_advice = self.openai_assistant.get_expert_advice(content)
                if expert_advice:
                    assistant_response = expert_advice
                else:
                    assistant_response = "I'm sorry, I couldn't find any information on that topic."

            # Combine the response and citations
            if citations:
                final_response = f"{assistant_response}\n\n" + "\n".join(citations)
            else:
                final_response = assistant_response

            logging.info(f"Final response prepared with {len(citations)} citations.")

            return {'role': 'assistant', 'content': final_response}

        except Exception as e:
            logging.error(f"Error processing message in LibrarianAgent: {str(e)}")
            return {'role': 'assistant', 'content': f"Error retrieving information: {str(e)}"}

# -----------------------------------
# Agent Factory
# -----------------------------------

class AgentFactory:
    """Factory to create different Assistant Agents"""

    @staticmethod
    def create_assistant_agent(config: Config) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="assistant",
            llm_config=config.llm_config,
            system_message=(
                "You are a central AI coordinator overseeing a collaborative multi-agent analysis. "
                "Facilitate seamless cooperation among agents, ensuring each one’s findings are shared, referenced, "
                "and built upon. Summarize key insights from each agent, direct questions to relevant agents, and "
                "reinforce a unified approach toward achieving insightful results aligned with project objectives."
            )
        )

    @staticmethod
    def create_librarian_agent(config: Config, openai_assistant: LibrarianAgent) -> autogen.AssistantAgent:
        return LibrarianAgent(
            name="librarian 1337",
            llm_config=config.llm_config,
            system_message=(
                "You are an expert research librarian with privileged access to scholarly resources and industry-grade knowledge. "
                "Collaborate by providing detailed, well-supported insights, citing any reliable resources for further verification. "
                "Highlight information that may assist other agents in their analyses, such as contextual knowledge for market trends, "
                "technical background for the tech expert, or foundational data for business strategy."
            ),
            openai_assistant=openai_assistant,
            assistant_id=config.ASSISTANT_ID,
            vector_store_ids=config.VECTOR_STORE_IDS
        )

    @staticmethod
    def create_web_researcher(config: Config, scraper: WebScraper) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="web_researcher",
            llm_config=config.llm_config,
            system_message=(
                "You are a web research specialist focused on gathering reliable information from authoritative sources. "
                "Coordinate with the Keyword Analyst to target specific topics and trends, and validate insights relevant to "
                "the Market Researcher and Business Consultant. Actively share links, summaries, or notable findings with other agents "
                "to enhance their understanding and analysis."
            )
        )

    @staticmethod
    def create_keyword_analyst(config: Config, analyzer: KeywordAnalyzer) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="keyword_analyst",
            llm_config=config.llm_config,
            system_message=(
                "You are a specialist in keyword extraction and intent analysis. Extract impactful themes and user intent, "
                "and share these keywords with the Web Researcher and Market Researcher to refine their queries and focus areas. "
                "Identify any emerging trends or patterns that could benefit the Business Consultant and Tech Expert, "
                "highlighting areas of strategic importance within the data."
            )
        )

    @staticmethod
    def create_market_researcher(config: Config) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="market_researcher",
            llm_config=config.llm_config,
            system_message=(
                "You are an advanced market researcher with expertise in understanding market dynamics and competitive landscapes. "
                "Use insights from the Web Researcher and Keyword Analyst to analyze customer segments, market trends, and competitive advantages. "
                "Share key market data with the Business Consultant to support financial modeling and with the Tech Expert if technology needs "
                "are impacted by market conditions."
            )
        )

    @staticmethod
    def create_tech_expert(config: Config) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="tech_expert",
            llm_config=config.llm_config,
            system_message=(
                "You are a technology strategist focused on evaluating technical feasibility and strategic alignment. "
                "Collaborate by reviewing findings from the Web Researcher and Keyword Analyst for any technical trends and potential constraints. "
                "Align your assessments with the Business Consultant to ensure the technology strategy supports profitability, scalability, "
                "and risk management goals. Provide specific, actionable recommendations, addressing any dependencies with other agents."
            )
        )

    @staticmethod
    def create_business_consultant(config: Config) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="business_consultant",
            llm_config=config.llm_config,
            system_message=(
                "You are a senior business consultant with expertise in strategy and financial modeling. "
                "Use insights from the Market Researcher and Tech Expert to create data-driven recommendations and realistic timelines. "
                "Consult the Librarian and Web Researcher to confirm assumptions or expand on any financial, market, or technical data. "
                "Provide a clear business strategy, sharing findings and collaborating closely with other agents to ensure holistic recommendations."
            )
        )

    @staticmethod
    def create_user_proxy(config: Config) -> autogen.UserProxyAgent:
        return autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=20,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("**FINAL ANALYSIS COMPLETE**"),
            code_execution_config={"work_dir": "analysis", "use_docker": False},
            llm_config=config.llm_config,
        )

# -----------------------------------
# Business Analyzer
# -----------------------------------

class BusinessAnalyzer:
    """Handles multi-agent analysis using AutoGen"""

    def __init__(self, config: Config):
        self.config = config
        self.openai_assistant = OpenAIAssistant(config)
        self.web_scraper = WebScraper(config)
        self.keyword_analyzer = KeywordAnalyzer()
        self.setup_agents()
        # Generate log filename
        self.log_file_path = generate_log_filename()

    def setup_agents(self):
        """Initialize the agent group"""
        logging.info("Setting up agents...")
        self.assistant = AgentFactory.create_assistant_agent(self.config)
        self.librarian = AgentFactory.create_librarian_agent(self.config, self.openai_assistant)
        self.web_researcher = AgentFactory.create_web_researcher(self.config, self.web_scraper)  # Ensure scraper is passed correctly
        self.keyword_analyst = AgentFactory.create_keyword_analyst(self.config, self.keyword_analyzer)
        self.marketer = AgentFactory.create_market_researcher(self.config)
        self.technologist = AgentFactory.create_tech_expert(self.config)
        self.business_consultant = AgentFactory.create_business_consultant(self.config)
        self.user_proxy = AgentFactory.create_user_proxy(self.config)

    def log_message(self, message: str):
        """Logs a message to the log file."""
        with open(self.log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(message + '\n')

    def gather_web_articles(self, project_brief: str, limit: int = 5) -> List[str]:
        """
        Gathers web articles related to the project brief by analyzing keywords
        and performing a refined search based on extracted themes.

        Args:
            project_brief (str): The main description of the project.
            limit (int): Maximum number of articles to gather per keyword.

        Returns:
            List[str]: A collection of articles related to the project brief.
        """
        logging.info(f"Starting keyword analysis for project brief: '{project_brief}'")

        # Step 1: Initialize KeywordAnalyzer and extract keywords from project brief
        keyword_analyzer = KeywordAnalyzer()
        extracted_keywords = keyword_analyzer.extract_keywords(project_brief)
        logging.info(f"Extracted keywords for web search: {extracted_keywords}")
        self.log_message(f"Extracted Keywords: {', '.join(extracted_keywords)}")

        # Step 2: Use extracted keywords to gather relevant articles
        articles = []
        for keyword in extracted_keywords:
            logging.info(f"Gathering articles for keyword: '{keyword}'")
        
            # Perform search with web scraper for each keyword
            keyword_articles = self.web_scraper.gather_articles(keyword, limit=limit)
            articles.extend(keyword_articles)  # Add keyword-specific articles to main list

            logging.info(f"Collected {len(keyword_articles)} articles for keyword '{keyword}'")

        # Step 3: Summarize total articles gathered
        total_articles = len(articles)
        logging.info(f"Total articles gathered: {total_articles}")
        self.log_message(f"Total Web Articles Collected: {total_articles}")

        return articles

    def _log_analysis_start(self, system_context: str, initial_prompt: str):
        """
        Logs the start of the analysis process including system context and initial prompt.
    
        Args:
            system_context (str): The system context being used for the analysis
            initial_prompt (str): The initial prompt to start the analysis
        """
        # Log analysis start with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
        # Log to system logger
        logging.info(f"Starting new analysis at {timestamp}")
    
        # Create formatted log message
        log_content = f"""
        === Analysis Started at {timestamp} ===

        System Context:
        {system_context}

        Initial Prompt:
        {initial_prompt}

        ===============================
        """
    
        # Use existing log_message method to write to file
        self.log_message(log_content)
    
        # Additional logging for significant parameters
        logging.info(f"Analysis parameters: Model={self.config.MODEL}, Temperature={self.config.TEMPERATURE}, Max Rounds={self.config.MAX_ROUND}")

    def create_analysis_prompt(self, project_brief: str) -> str:
        """Creates the initial analysis prompt"""
        # Gather web data
        web_articles = self.gather_web_articles(project_brief, limit=5)
        concatenated_articles = "\n\n".join(web_articles)

        # Get expert insights from Librarian Agent using file_search
        librarian_response = self.librarian.process_message({'content': f'Drill into expert business advice on negotiation, change management, finances and marketing for: {project_brief}'})
        expert_advice = librarian_response.get('content', 'No expert advice available.')

        # Log expert advice
        self.log_message(f"Expert Advice:\n{expert_advice}\n")

        # Analyze keywords and intent
        keywords = self.keyword_analyzer.extract_keywords(project_brief)
        intent_analysis = self.keyword_analyzer.analyze_intent(project_brief)

        # Log keyword analysis
        self.log_message(f"Keywords Extracted: {keywords}")
        self.log_message(f"Intent Analysis: {intent_analysis}\n")

        # Prepare web articles summary
        web_articles_summary = "\n\n".join([
            f"Article {i+1}: {article[:1000]}..." for i, article in enumerate(web_articles[:30])
        ]) if web_articles else "No web articles found."

        self.log_message(f"Web Articles Summary:\n{web_articles_summary}\n")

        initial_prompt = f"""
        Analyze this business opportunity: {project_brief}

        **Keywords:** {', '.join(keywords)}
        **Top Themes:** {', '.join(f"{word}: {count}" for word, count in intent_analysis['top_themes'])}

        **Instructions for Agents:**
        - Each agent should perform their designated tasks as outlined below.
        - Share your findings with other agents to build upon each other's work.
        - Respectfully challenge or question findings from others if you have evidence or reasoning to support your viewpoint.
        - Ensure that all analyses are data-driven and well-supported.
        - Review the findings from previous messages by other agents.
        - Integrate insights from others to refine their own responses.
        - Raise questions or points of clarification if any data or insights seem unclear.
        - Provide supportive evidence when challenging others' analyses.

        **Agent Tasks:**
        1. **Keyword Analyst:**
           - Analyze key themes and concepts.
           - Identify market-relevant keywords.
           - Determine user intent and needs.
           
        2. **Web Researcher:**
           - Gather at least 20 current, relevant sources from Wikipedia, Forbes, McKinsey, Fortune, Financial Times, and Wall Street Journal.
           - Analyze online trends.
           - Research competitors.
           - Provide summaries of findings.

        3. **Librarian:**
           - Consult expert knowledge using the OpenAI Assistant with File Search.
           - Provide research-backed insights.
           - Recommend best practices.
           - Challenge any assumptions made by other agents with supporting evidence.

        4. **Market Researcher:**
           - Analyze market size and potential.
           - Identify customer segments and needs.
           - Conduct competition analysis.
           - Suggest marketing strategies.
           - Explore growth opportunities.

        5. **Tech Expert:**
           - Assess technical requirements.
           - Outline the development process.
           - Identify infrastructure needs.
           - Propose quality control measures.
           - Evaluate technical risks.

        6. **Business Consultant:**
           - Prepare financial projections.
           - Determine resource requirements.
           - Develop an implementation timeline.
           - Conduct a risk assessment.
           - Provide strategic recommendations.


        **Final Deliverable:**
        - After completing your analyses and discussions, collaborate to create a final summary of key findings and recommendations that aims to give comprehensive overview of actions and activities needed.
        - The last message from agent assistant shall conclude with "**FINAL ANALYSIS COMPLETE**"
        """

        system_context =f"""
        This is specific context, expert insights for performance of the task (extra contextual background)

        **Expert Insights:**
        {expert_advice}

        **Web Articles Summary:**
        {web_articles_summary}

        """

        return {"initial_prompt": initial_prompt, "system_context": system_context}

    def run_analysis(self, project_brief: str, model: str = None, temperature: float = None, max_round: int = None) -> str:
        """Runs the multi-agent analysis"""
        logging.info("Running analysis...")
        try:
            # Update config if customization is provided
            if model:
                self.config.MODEL = model
                self.config.llm_config["config_list"][0]["model"] = model
            if temperature is not None:
                self.config.TEMPERATURE = temperature
                self.config.llm_config["temperature"] = temperature
            max_round = int(max_round) if max_round else self.config.MAX_ROUND
            
            # Get initial prompt and system context from create_analysis_prompt
            prompts = self.create_analysis_prompt(project_brief)
            initial_prompt = prompts["initial_prompt"]
            system_context = prompts["system_context"]

            # Create the chat group with all agents
            groupchat = autogen.GroupChat(
                agents=[
                    self.user_proxy, self.assistant,
                    self.web_researcher, self.keyword_analyst,
                    self.marketer, self.technologist, self.business_consultant
                ],
                messages=[{"role": "system", "content": system_context}],
                max_round=max_round
            )

            # Create the group chat manager
            manager = autogen.GroupChatManager(
                groupchat=groupchat,
                llm_config=self.config.llm_config
            )


            # Start the analysis
            logging.debug(f"Initial Prompt: {system_context}")
            logging.debug(f"Initial Prompt: {initial_prompt}")

            # Log the initial prompt
            self.log_message(f"System Prompt:\n{system_context}\n")
            self.log_message(f"Initial Prompt:\n{initial_prompt}\n")

            self.user_proxy.initiate_chat(
                manager,
                message=initial_prompt
            )

            # Extract the analysis results
            results = []
            for message in groupchat.messages:
                # Check if message is a dictionary, and if it’s a list, handle appropriately
                if isinstance(message, dict):
                    # Proceed as a dictionary
                    role = message.get("role", "")
                    name = message.get("name", "unknown")
                    content = message.get("content", "")
                elif isinstance(message, list):
                    # Log or handle unexpected list structure in messages
                    logging.error(f"Unexpected list structure in messages: {message}")
                    continue  # Skip or handle as needed
                else:
                    # Handle as an object or fallback
                    role = getattr(message, 'role', '')
                    name = getattr(message, 'name', 'unknown')
                    content = getattr(message, 'content', '')

                # Process message based on role
                if role != "user_proxy":
                    results.append(f"**{name.capitalize()}**:\n{content}\n")

            final_result = "\n".join(results)

            # Log all messages in the group chat
            self.log_message("Group Chat Messages:")
            for message in groupchat.messages:
                if isinstance(message, dict):
                    name = message.get("name", "unknown")
                    role = message.get("role", "")
                    content = message.get("content", "")
                    timestamp = message.get("timestamp", "")
                else:
                    name = getattr(message, 'name', 'unknown')
                    role = getattr(message, 'role', '')
                    content = getattr(message, 'content', '')
                    timestamp = getattr(message, 'timestamp', '')

                self.log_message(f"[{timestamp}] {role} ({name}): {content}")

            # Log the final result
            self.log_message(f"Final Result:\n{final_result}\n")

            # Sanitize the HTML to prevent XSS
            ALLOWED_TAGS = list(bleach.sanitizer.ALLOWED_TAGS) + [
                'p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
            ]
            ALLOWED_ATTRIBUTES = dict(bleach.sanitizer.ALLOWED_ATTRIBUTES)
            final_result = bleach.clean(final_result, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES)

            logging.info("Analysis completed successfully.")
            return final_result

        except Exception as e:
            error_message = f"Error during analysis: {str(e)}"
            logging.error(error_message)
            self.log_message(error_message)
            return error_message



# -----------------------------------
# Flask Application Setup
# -----------------------------------

app = Flask(__name__)

# Initialize configuration
config = Config()
app.config['SECRET_KEY'] = config.SECRET_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------
# HTML Template
# -----------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Autogen OpenAI Swarm - Business Analyzer</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css">
    <style>
        body {
            background-color: #f8f9fa;
        }
        h1, h2, h4 {
            text-align: center;
        }
        #result {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 5px;
            white-space: pre-wrap; /* Wrap long lines */
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4">Business Analyzer</h1>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, message in messages %}
              <div class="alert alert-{{ category }}" role="alert">
                {{ message }}
              </div>
            {% endfor %}
          {% endif %}
        {% endwith %}
        
        <form method="POST">
            <div class="form-group">
                <label for="project_brief">Project Brief</label>
                <textarea class="form-control" id="project_brief" name="project_brief" rows="5" placeholder="Describe your business opportunity...">{{ project_brief }}</textarea>
            </div>
            
            <hr>
            
            <h4>Configuration Settings</h4>
            <div class="form-row">
                <div class="form-group col-md-4">
                    <label for="model">Model</label>
                    <input type="text" class="form-control" id="model" name="model" placeholder="e.g., gpt-4o-mini" value="{{ model }}">
                </div>
                <div class="form-group col-md-4">
                    <label for="temperature">Temperature</label>
                    <input type="number" step="0.1" min="0" max="1" class="form-control" id="temperature" name="temperature" placeholder="e.g., 0.1" value="{{ temperature }}">
                </div>
                <div class="form-group col-md-4">
                    <label for="max_round">Max Rounds</label>
                    <input type="number" min="1" class="form-control" id="max_round" name="max_round" placeholder="e.g., 10" value="{{ max_round }}">
                </div>
            </div>
            
            <button type="submit" class="btn btn-primary">Run Analysis</button>
        </form>
        
        {% if result %}
            <hr>
            <h2>Analysis Results</h2>
            <div id="result">{{ result | safe }}</div>
        {% endif %}
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
</body>
</html>
"""

# -----------------------------------
# Flask Routes
# -----------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data
        project_brief = request.form.get('project_brief', '').strip()
        model = request.form.get('model', '').strip()
        temperature = request.form.get('temperature', '').strip()
        max_round = request.form.get('max_round', '').strip()

        # Validate project brief
        if not project_brief:
            flash('Project brief cannot be empty.', 'warning')
            return render_template_string(HTML_TEMPLATE, project_brief=project_brief, model=model, temperature=temperature, max_round=max_round)
        
        # Initialize configuration with possible overrides from UI
        current_config = Config()
        if model:
            current_config.MODEL = model
        if temperature:
            try:
                current_config.TEMPERATURE = float(temperature)
            except ValueError:
                flash('Temperature must be a number between 0 and 1.', 'danger')
                return render_template_string(HTML_TEMPLATE, 
                                              project_brief=project_brief, 
                                              model=model,
                                              temperature=temperature,
                                              max_round=max_round)
        if max_round:
            try:
                current_config.MAX_ROUND = int(max_round)
            except ValueError:
                flash('Max Round must be an integer.', 'danger')
                return render_template_string(HTML_TEMPLATE, 
                                              project_brief=project_brief, 
                                              model=model,
                                              temperature=temperature,
                                              max_round=max_round)
        
        # Create analyzer
        analyzer = BusinessAnalyzer(current_config)
        result = analyzer.run_analysis(project_brief, model=model, temperature=current_config.TEMPERATURE, max_round=current_config.MAX_ROUND)
        
        if result.startswith("Error"):
            flash(result, 'danger')
        else:
            flash('Analysis completed successfully!', 'success')
        
        return render_template_string(HTML_TEMPLATE, 
                                      project_brief=project_brief, 
                                      model=model,
                                      temperature=temperature,
                                      max_round=max_round,
                                      result=result)
    
    # GET request
    return render_template_string(HTML_TEMPLATE)

# -----------------------------------
# Entry Point
# -----------------------------------

if __name__ == "__main__":
    # Ensure that the necessary environment variables are set
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == 'your-default-openai-api-key':
        logger.warning("OPENAI_API_KEY is not set. Please set it as an environment variable.")
    
    if not config.SECRET_KEY or config.SECRET_KEY == 'your-secret-key':
        logger.warning("SECRET_KEY is not set. Please set it as an environment variable for better security.")
    
    if not config.ASSISTANT_ID or config.ASSISTANT_ID == 'asst_a2rbz1zXNB691acwLF9vsCng':
        logger.warning("ASSISTANT_ID is not set. Please set it as an environment variable.")
    
    if not config.VECTOR_STORE_IDS or config.VECTOR_STORE_IDS == ['']:
        logger.warning("VECTOR_STORE_IDS are not set. Please set them as environment variables separated by commas.")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
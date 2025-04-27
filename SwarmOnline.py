import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from flask import Flask, request, render_template_string, flash
import autogen
import bleach
import markdown

# -----------------------------------
# Configuration Settings
# -----------------------------------

@dataclass
class Config:
    """Configuration settings"""
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', 'insert-api-key-here')
    MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.7
    MAX_ROUND: int = 10
    SECRET_KEY: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'secret-key'))
    ASSISTANT_ID: str = field(default_factory=lambda: os.getenv('ASSISTANT_ID', ''))
    
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
# Web Scraping
# -----------------------------------

class WebScraper:
    """Simplified web scraping for content extraction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def scrape_url(self, url: str) -> str:
        """Basic URL scraping with minimal processing"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for tag in ['script', 'style', 'nav', 'header', 'footer']:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Extract text from paragraphs
            paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if text and len(text) > 100:  # Basic length filter
                    paragraphs.append(text)
            
            return ' '.join(paragraphs[:3])  # Return first 3 substantial paragraphs
            
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return ""


# -----------------------------------
# OpenAI Assistant Integration
# -----------------------------------

class OpenAIAssistant:
    """Simplified OpenAI Assistant interaction"""

    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.assistant_id = config.ASSISTANT_ID

    def get_response(self, query: str) -> str:
        """Get a response from the OpenAI Assistant"""
        try:
            thread = self.client.beta.threads.create()
            
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )

            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )

            # Wait for completion
            while run.status in ["queued", "in_progress"]:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            
            for msg in messages.data:
                if msg.role == 'assistant':
                    return msg.content[0].text.value

            return "No response available."

        except Exception as e:
            logging.error(f"Error getting response: {str(e)}")
            return f"Error: {str(e)}"


# -----------------------------------
# Business Analyzer
# -----------------------------------

class BusinessAnalyzer:
    """Enhanced business analysis using multiple specialized agents"""

    def __init__(self, config: Config):
        self.config = config
        self.openai_assistant = OpenAIAssistant(config)
        self.web_scraper = WebScraper()
        self.setup_agents()

    def setup_agents(self):
        """Initialize specialized agents"""
        # Coordinator Agent
        self.coordinator = autogen.AssistantAgent(
            name="coordinator",
            llm_config=self.config.llm_config,
            system_message="""You are the analysis coordinator. Your role is to:
                1. Guide the conversation flow
                2. Ensure all aspects are covered
                3. Synthesize insights from other agents
                4. Request specific input when needed
                5. Prepare the final summary
            """
        )
        
        # Market Analysis Agent
        self.market_analyst = autogen.AssistantAgent(
            name="market_analyst",
            llm_config=self.config.llm_config,
            system_message="""You are a market analysis expert. Focus on:
                1. Market size and growth potential
                2. Customer segmentation
                3. Competitive analysis
                4. Industry trends
                5. Market entry strategies
            """
        )
        
        # Technical Expert
        self.tech_expert = autogen.AssistantAgent(
            name="tech_expert",
            llm_config=self.config.llm_config,
            system_message="""You are a technical expert. Analyze:
                1. Technical requirements
                2. Implementation challenges
                3. Resource needs
                4. Technology stack recommendations
                5. Technical risk assessment
            """
        )
        
        # Financial Advisor
        self.financial_advisor = autogen.AssistantAgent(
            name="financial_advisor",
            llm_config=self.config.llm_config,
            system_message="""You are a financial advisor. Provide insights on:
                1. Revenue models
                2. Cost structure
                3. Investment requirements
                4. Financial projections
                5. Risk management
            """
        )

        # User Proxy (without Docker)
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            llm_config=self.config.llm_config,
            code_execution_config={
                "work_dir": "analysis",
                "use_docker": False
            }
        )

    def run_analysis(self, project_brief: str) -> str:
        """Run enhanced multi-agent analysis"""
        try:
            # Get initial insights
            expert_insights = self.openai_assistant.get_response(project_brief)
            
            # Create analysis prompt
            prompt = f"""
            Project Brief: {project_brief}

            Expert Insights: {expert_insights}

            Instructions for Analysis Team:

            1. Market Analyst:
               - Analyze market size and potential
               - Identify target customer segments
               - Assess competition
               - Evaluate market trends
               - Suggest positioning strategy

            2. Technical Expert:
               - Evaluate technical feasibility
               - Outline implementation requirements
               - Identify potential technical challenges
               - Recommend technology stack
               - Assess scalability concerns

            3. Financial Advisor:
               - Propose revenue model
               - Estimate initial costs
               - Project financial metrics
               - Identify funding requirements
               - Assess financial risks

            4. Coordinator:
               - Guide the discussion
               - Request clarification when needed
               - Ensure comprehensive coverage
               - Synthesize insights
               - Prepare final recommendations

            Each agent should:
            - Provide specific, actionable insights
            - Support conclusions with reasoning
            - Consider interdependencies with other aspects
            - Highlight risks and opportunities
            - Suggest concrete next steps

            The discussion should conclude with a synthesized set of recommendations.
            """

            # Create enhanced group chat
            groupchat = autogen.GroupChat(
                agents=[
                    self.user_proxy,
                    self.coordinator,
                    self.market_analyst,
                    self.tech_expert,
                    self.financial_advisor
                ],
                messages=[],
                max_round=self.config.MAX_ROUND
            )

            manager = autogen.GroupChatManager(
                groupchat=groupchat,
                llm_config=self.config.llm_config
            )

            # Run analysis
            self.user_proxy.initiate_chat(manager, message=prompt)

            # Extract and format results
            results = []
            current_section = None
            
            for message in groupchat.messages:
                if isinstance(message, dict) and message.get("role") != "user_proxy":
                    name = message.get('name', '').replace('_', ' ').title()
                    content = message.get('content', '').strip()
                    
                    if content:
                        # Add section header if agent changes
                        if current_section != name:
                            results.append(f"\n## {name} Analysis")
                            current_section = name
                        
                        # Add content with formatting
                        results.append(f"{content}\n")

            final_analysis = "\n".join(results)
            
            # Clean and format the output
            cleaned_analysis = bleach.clean(
                final_analysis,
                tags=['h1', 'h2', 'h3', 'p', 'br', 'strong', 'em', 'ul', 'ol', 'li'],
                strip=True
            )

            return cleaned_analysis if cleaned_analysis else "No analysis results generated."

        except Exception as e:
            error_message = f"Error during analysis: {str(e)}"
            logging.error(error_message)
            return error_message


# -----------------------------------
# Flask Application
# -----------------------------------

app = Flask(__name__)
config = Config()
app.config['SECRET_KEY'] = config.SECRET_KEY

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Business Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            line-height: 1.6;
        }
        .analysis-result {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .analysis-result h1,
        .analysis-result h2,
        .analysis-result h3,
        .analysis-result h4 {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .analysis-result ul,
        .analysis-result ol {
            padding-left: 1.5rem;
            margin-bottom: 1rem;
        }
        .analysis-result li {
            margin-bottom: 0.5rem;
        }
        .analysis-result code {
            background-color: #f8f9fa;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.875em;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <h1 class="text-center mb-4">Business Analyzer</h1>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="card mb-4">
            <div class="card-body">
                <form method="POST">
                    <div class="mb-3">
                        <label class="form-label">Project Brief</label>
                        <textarea class="form-control" name="project_brief" rows="5" required>{{ project_brief }}</textarea>
                    </div>
                    <button type="submit" class="btn btn-primary">Analyze</button>
                </form>
            </div>
        </div>
        
        {% if result %}
            <div class="analysis-result">
                {{ result | safe }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    project_brief = ""
    
    if request.method == 'POST':
        project_brief = request.form.get('project_brief', '').strip()
        
        if not project_brief:
            flash('Please provide a project brief.', 'warning')
            return render_template_string(HTML_TEMPLATE)
        
        analyzer = BusinessAnalyzer(config)
        markdown_result = analyzer.run_analysis(project_brief)
        
        # Convert markdown to HTML using python-markdown
        html_result = markdown.markdown(markdown_result, extensions=['tables', 'fenced_code'])
        
        return render_template_string(
            HTML_TEMPLATE, 
            project_brief=project_brief,
            result=html_result
        )
    
    return render_template_string(HTML_TEMPLATE, project_brief=project_brief)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)

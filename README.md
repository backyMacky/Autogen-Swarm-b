# Autogen Swarm Business Analyzer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?logo=OpenAI&logoColor=white)](https://openai.com)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![GitHub issues](https://img.shields.io/github/issues/backyMacky/Autogen-Swarm-b)](https://github.com/backyMacky/Autogen-Swarm-b/issues)
[![GitHub stars](https://img.shields.io/github/stars/backyMacky/Autogen-Swarm-b)](https://github.com/backyMacky/Autogen-Swarm-b/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/backyMacky/Autogen-Swarm-b)](https://github.com/backyMacky/Autogen-Swarm-b/network/members)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/backyMacky/Autogen-Swarm-b/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A "sophisticated" business analysis tool that leverages multiple AI agents to provide comprehensive business insights. The system uses OpenAI's GPT models and AutoGen framework to simulate a team of business experts analyzing your project or business idea.

Live Demo: [[Demo link](http://77.160.195.198:88/)]

<a href="https://www.buymeacoffee.com/bloombrine" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

## Features

- 🤖 **Multi-Agent Analysis System**
  - Market Analysis Expert
  - Technical Expert
  - Financial Advisor
  - Coordination Agent

- 🧠 **Comprehensive Analysis Coverage**
  - Market size and potential
  - Technical feasibility
  - Financial projections
  - Risk assessment

- 🌐 **Web Interface**
  - Clean, responsive design
  - Easy project brief submission
  - Formatted analysis results
  - Bootstrap-based UI

## Prerequisites

- Python 3.9+
- OpenAI API key
- OpenAI Assistant ID

## Architecture
```mermaid
classDiagram
    class Config {
        +OPENAI_API_KEY: str
        +MODEL: str
        +TEMPERATURE: float
        +MAX_ROUND: int
        +SECRET_KEY: str
        +ASSISTANT_ID: str
        +llm_config(): Dict
    }

    class WebScraper {
        -session: Session
        +scrape_url(url: str): str
    }

    class OpenAIAssistant {
        -client: OpenAI
        -assistant_id: str
        +get_response(query: str): str
    }

    class BusinessAnalyzer {
        -config: Config
        -openai_assistant: OpenAIAssistant
        -web_scraper: WebScraper
        -coordinator: AssistantAgent
        -market_analyst: AssistantAgent
        -tech_expert: AssistantAgent
        -financial_advisor: AssistantAgent
        -user_proxy: UserProxyAgent
        +setup_agents()
        +run_analysis(project_brief: str): str
    }

    class FlaskApp {
        +config: Config
        +HTML_TEMPLATE: str
        +index(): str
    }

    class AssistantAgent {
        +name: str
        +llm_config: Dict
        +system_message: str
    }

    class UserProxyAgent {
        +name: str
        +human_input_mode: str
        +max_consecutive_auto_reply: int
        +llm_config: Dict
        +code_execution_config: Dict
    }

    BusinessAnalyzer --> Config
    BusinessAnalyzer --> OpenAIAssistant
    BusinessAnalyzer --> WebScraper
    BusinessAnalyzer --> AssistantAgent
    BusinessAnalyzer --> UserProxyAgent
    FlaskApp --> BusinessAnalyzer
    FlaskApp --> Config
    OpenAIAssistant --> Config
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/backyMacky/Autogen-Swarm-b.git
cd Autogen-Swarm-b
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install required packages:
```bash
pip install flask openai autogen beautifulsoup4 requests markdown bleach
```

4. Set up your OpenAI API key and Assistant ID in `Config` class

5. Run the application:
```bash
python SwarmOnline.py
```

6. Open your browser and navigate to:
```
http://localhost:5000
```

## Configuration

Update the `Config` class in SwarmOnline.py:
```python
@dataclass
class Config:
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
    MODEL: str = "gpt-4o-mini"  # Your preferred model
    TEMPERATURE: float = 0.7    
    MAX_ROUND: int = 10        
    SECRET_KEY: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'secret-key'))
    ASSISTANT_ID: str = field(default_factory=lambda: os.getenv('ASSISTANT_ID', ''))
```

## How It Works

1. **Input**: Submit your project brief through the web interface
2. **Analysis**: Multiple AI agents analyze different aspects:
   - Market Analysis
   - Technical Assessment
   - Financial Analysis
   - Coordinated Insights
3. **Output**: Receive a comprehensive analysis in formatted Markdown

## Process Flow
```mermaid
graph TB
    subgraph "Web Layer"
        A[Flask Web Interface]
    end

    subgraph "Business Logic Layer"
        B[Business Analyzer]
        C[OpenAI Assistant]
        D[Web Scraper]
    end

    subgraph "Agent Layer"
        E[Coordinator Agent]
        F[Market Analyst Agent]
        G[Technical Expert Agent]
        H[Financial Advisor Agent]
        I[User Proxy Agent]
    end

    subgraph "External Services"
        J[OpenAI API]
        K[Web Resources]
    end

    A --> B
    B --> C
    B --> D
    B --> E
    E --> F
    E --> G
    E --> H
    E --> I
    C --> J
    D --> K

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#ddf,stroke:#333,stroke-width:2px
    style D fill:#ddf,stroke:#333,stroke-width:2px
    style E fill:#dfd,stroke:#333,stroke-width:2px
    style F fill:#dfd,stroke:#333,stroke-width:2px
    style G fill:#dfd,stroke:#333,stroke-width:2px
    style H fill:#dfd,stroke:#333,stroke-width:2px
    style I fill:#dfd,stroke:#333,stroke-width:2px
    style J fill:#fdd,stroke:#333,stroke-width:2px
    style K fill:#fdd,stroke:#333,stroke-width:2px
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation and Tutorials

- [AutoGen Documentation](https://microsoft.github.io/autogen/0.2/docs/Getting-Started/)
- [Flask Documentation](https://flask.palletsprojects.com/en/2.3.x/)
- [Bootstrap Documentation](https://getbootstrap.com/docs/5.3/getting-started/introduction/)

## Related Projects

- [OpenAI's Swarm](https://github.com/openai/swarm)
- [AgentScan: Multi-Agent Observability](https://github.com/The-Swarm-Corporation/agentscan)

## Community and Support

- [AutoGen Discord Community](https://discord.gg/pAbnFJrkgZ)
- [Flask Mailing List](https://mail.python.org/mailman/listinfo/flask)

## Acknowledgments

- [OpenAI](https://openai.com) for GPT models
- [Microsoft AutoGen Framework](https://microsoft.github.io/autogen/0.2/)
- [Flask Framework](https://flask.palletsprojects.com/)
- [Bootstrap](https://getbootstrap.com/) for UI components

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/backyMacky/Autogen-Swarm-b/issues)

---

Made with ❤️ by [Martin Bacigal](https://github.com/backyMacky)

"""
LangChain Blog Generator using Groq and LLama 3.1
A professional Blog Generator that leverages Wikipedia and Google search for intelligent responses.

Author: Emmanuel Efienemokwu
Date: 2025
"""


import os
import sys
import logging
import traceback
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from groq import BadRequestError

# ============================================================
# 🧩 1. Configure Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 🔧 2. Load Environment Variables
# ============================================================
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    logger.critical("❌ GROQ_API_KEY not found in environment. Please set it in your .env file.")
    sys.exit(1)

# ============================================================
#  3. Initialize Tools
# ============================================================
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
web_search_tool = DuckDuckGoSearchRun()
tools = [wikipedia_tool, web_search_tool]

# ============================================================
#  4. Preferred Models (with fallback order)
# ============================================================
PREFERRED_MODELS = [
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

# ============================================================
#  5. Initialize LLM with Fallback and Error Handling
# ============================================================
def initialize_llm_with_fallback(api_key: str, preferred_models=PREFERRED_MODELS, **kwargs):
    """
    Try initializing the LLM with multiple fallback models.
    Raises RuntimeError if all models fail.
    """
    last_exception = None
    for model_name in preferred_models:
        logger.info(f"🧠 Attempting to initialize model: {model_name}")
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                streaming=True,
                temperature=0.3,
                **kwargs,
            )
            logger.info(f"✅ Successfully initialized model: {model_name}")
            return llm

        except BadRequestError as e:
            last_exception = e
            logger.warning(f"⚠️ Model '{model_name}' unavailable or decommissioned: {e}")
        except Exception as e:
            last_exception = e
            logger.error(f"❗ Unexpected error while initializing '{model_name}': {e}")
            logger.debug(traceback.format_exc())

    # If no model succeeded
    raise RuntimeError(
        "❌ Failed to initialize any preferred LLM models.\n"
        "Please verify your GROQ_API_KEY and check available models in your Groq dashboard."
    ) from last_exception

# ============================================================
# 🤖 6. Initialize Agent Safely
# ============================================================
try:
    llm = initialize_llm_with_fallback(api_key, preferred_models=PREFERRED_MODELS)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    logger.info("🤖 Agent successfully initialized.\n")

except Exception as e:
    logger.critical("🔥 Critical error initializing the agent.")
    logger.debug(traceback.format_exc())
    sys.exit(1)

# ============================================================
# ✍️ 7. Blog Generation Function
# ============================================================
def generate_blog(topic: str) -> str:
    """
    Generate a structured blog for a given topic using the initialized agent.
    Handles API errors gracefully.
    """
    prompt = f"""
    You are an AI writing assistant.
    Create a well-structured blog post on the topic: "{topic}".
    Use reliable facts from research tools.
    Structure it as follows:

    ## Heading
    ## Introduction
    ## Content
    ## Summary
    """

    try:
        logger.info("🚀 Generating blog content...")
        response = agent.run(prompt)
        logger.info("✅ Blog successfully generated.")
        return response

    except BadRequestError as e:
        logger.error("Groq API returned a bad request — possible causes:")
        logger.error("- Model may be deprecated or invalid.")
        logger.error("- Invalid prompt or rate limit reached.")
        logger.error(f"Details: {e}")
        logger.debug(traceback.format_exc())

    except Exception as e:
        logger.error("Unexpected error during blog generation:")
        logger.debug(traceback.format_exc())

    # Return safe message on failure
    return (
        "⚠️ Sorry, something went wrong while generating the blog. "
        "Please check your connection or API key and try again."
    )

# ============================================================
# 🧭 8. CLI Entrypoint
# ============================================================
if __name__ == "__main__":
    try:
        topic = input("📝 Enter a blog topic: ").strip()
        if not topic:
            logger.error("❌ Topic cannot be empty.")
            sys.exit(1)

        blog = generate_blog(topic)

        print("\n===== 📰 Generated Blog =====\n")
        print(blog)

    except KeyboardInterrupt:
        logger.info("👋 Exiting gracefully. Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.critical("💀 Unexpected fatal error in main execution.")
        logger.debug(traceback.format_exc())
        sys.exit(1)
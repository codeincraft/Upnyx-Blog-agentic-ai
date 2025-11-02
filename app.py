"""
LangChain Blog Generator using Groq and LLama 3.1
A professional Blog Generator that leverages Wikipedia and Google search for intelligent responses.

Author: Emmanuel Efienemokwu
Date: 2025
"""



import os
import sys
import traceback
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from groq import BadRequestError


# ============================================================
# 🔧 1. Load environment variables
# ============================================================
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print(" Error: GROQ_API_KEY not found in environment. Please set it in your .env file.")
    sys.exit(1)


# ============================================================
#  2. Initialize tools
# ============================================================
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
web_search_tool = DuckDuckGoSearchRun()
tools = [wikipedia_tool, web_search_tool]


# ============================================================
#  3. Preferred models (with fallback order)
# ============================================================
PREFERRED_MODELS = [  
    "llama-3.1-8b-instant",     # original
    "mixtral-8x7b-32768",       # fallback
]


# ============================================================
#  4. Initialize LLM with Fallback and Error Handling
# ============================================================
def initialize_llm_with_fallback(api_key: str, preferred_models=PREFERRED_MODELS, **kwargs):
    """
    Try initializing the LLM with multiple fallback models.
    Raises RuntimeError if all models fail.
    """
    last_exception = None
    for model_name in preferred_models:
        print(f"🧠 Attempting to initialize model: {model_name}")
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                streaming=True,
                temperature=0.3,
                **kwargs,
            )
            print(f" Successfully initialized model: {model_name}")
            return llm

        except BadRequestError as e:
            last_exception = e
            print(f" Model '{model_name}' unavailable or decommissioned:\n   {e}")
        except Exception as e:
            last_exception = e
            print(f" Unexpected error while initializing '{model_name}': {e}")
            traceback.print_exc()

    # If no model succeeded
    raise RuntimeError(
        " Failed to initialize any preferred LLM models.\n"
        "Please verify your GROQ_API_KEY and check available models in your Groq dashboard."
    ) from last_exception


# ============================================================
# 🧩 5. Initialize Agent Safely
# ============================================================
try:
    llm = initialize_llm_with_fallback(api_key, preferred_models=PREFERRED_MODELS)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    print("🤖 Agent successfully initialized.\n")

except Exception as e:
    print(" Critical error initializing the agent:")
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# ✍️ 6. Blog Generation Function (with robust error handling)
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
        print("🚀 Generating blog content...")
        response = agent.run(prompt)
        print("✅ Blog successfully generated.")
        return response

    except BadRequestError as e:
        print(" Groq API returned a bad request. Possible causes:")
        print("- Model may be deprecated or invalid.")
        print("- Invalid prompt or rate limit reached.")
        print(f"Details: {e}")
        traceback.print_exc()

    except Exception as e:
        print(" Unexpected error during blog generation:")
        traceback.print_exc()

    # Return safe message on failure
    return (
        " Sorry, something went wrong while generating the blog. "
        "Please check your connection or API key and try again."
    )


# ============================================================
# 🧭 7. CLI Entrypoint
# ============================================================
if __name__ == "__main__":
    try:
        topic = input("📝 Enter a blog topic: ").strip()
        if not topic:
            print(" Error: Topic cannot be empty.")
            sys.exit(1)

        blog = generate_blog(topic)

        print("\n===== 📰 Generated Blog =====\n")
        print(blog)

    except KeyboardInterrupt:
        print("\n🚪 Exiting gracefully. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(" Unexpected fatal error in main execution:")
        traceback.print_exc()
        sys.exit(1)

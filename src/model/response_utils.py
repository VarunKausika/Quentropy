from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

@dataclass
class LLMResult:
    """LLM response to a given prompt"""
    result: str

@dataclass
class LLMSingleResult: 
    """LLM response to a single prompt"""
    result: str
    message_queue: list[dict]

@dataclass
class LLMMapResult:
    """LLM map response to a given prompt"""

    partial_message_queues: list[list[dict]]

@dataclass
class LLMMapReduceResult:
    """LLM map-reduce response to given prompt"""

    result: str
    partial_message_queues: list[list[dict]]

@dataclass
class LLMRefineResult:
    """LLM refine response to a given prompt"""

    result: str
    message_queue: list[dict]

SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner
Here are some rules you must always follow:
- Generate human readable output, avoid creating output with gibberish text
- Generate only the requested output, don't include any other alnguage before or after the reqeusted output
- Never say thank you, or that you are happy to help, or that you are an AI agent, etc. Just answer directly
- Generate professional language, which should never contain offensive or foul language
- You need to respond in less than {max_new_tokens} tokens
"""

SUPPORTED_MODELS = []

DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 0.92
DEFAULT_MAX_TOKENS = 256
DEFAULT_STREAM = False
DEFAULT_STOP_TOKENS = []
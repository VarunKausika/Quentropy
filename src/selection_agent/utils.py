from src.cand_gen.utils import get_schema
from .prompts import *
from src.model.inference_endpoints import LLM

def select_agent(
    ir: list[str],
    database_name: str,
    database_path: str,
    question: str,
    hint: str,
    queries: list[str],
    results: list[str],
    model: LLM
):
    DATABASE_SCHEMA = get_schema(
        ir = ir, 
        database_name = database_name,
        database_path = database_path
    )

    prompt = SA_PROMPT.format(
        db_schema = DATABASE_SCHEMA,
        question = question,
        hint = hint,
        query_1 = queries[0],
        result_1 = results[0],
        query_2 = queries[1],
        result_2 = results[1]
    )

    completion = model(
        response_type = 'single',
        prompts = prompt
    ).result

    return completion
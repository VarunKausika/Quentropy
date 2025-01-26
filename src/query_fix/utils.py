from src.cand_gen.utils import get_schema
from .prompts import *
from src.model.inference_endpoints import LLM

def query_fixer(
    database_name: str,
    database_root_path: str,
    question: str,
    hint: str,
    ir: list[str],
    query_to_correct: str,
    result: str,
    model: LLM
):

    db_schema = get_schema(
        ir=ir,
        database_name = database_name,
        database_path = database_root_path,
    )

    prompt = QF_PROMPT.format(
        QUESTION=question,
        HINT=hint,
        QUERY=query_to_correct,
        DATABASE_SCHEMA=db_schema,
        EXAMPLES=QF_EXAMPLES,
        RESULT=result
    )

    completion = model(
        response_type='single',
        prompts = prompt
    ).result

    return completion

def parse_query_fix_output(
    sql_query: str
):
    return sql_query.strip('```').lstrip('sql')
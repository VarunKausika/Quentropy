import re
from collections import defaultdict
import pandas as pd
from random import shuffle
from src.model.inference_endpoints import LLM
from .prompts import *

def get_schema(
    ir: list[str],
    database_path: str,
    database_name: str
):
    col_hash = defaultdict(list)
    for col in ir:
        parts = col.split('.')
        table = parts[0].strip('`')
        col = parts[1].strip('`')
        if len(parts) == 3:
            value = parts[-1].strip('`')
            col_hash[table].append((col, value))
        else:
            col_hash[table].append(col)
    
    tab_schema_strs = []
    for table_name, columns in col_hash.items():
        schema_df = pd.read_csv(f'{database_path}/database_description/{table_name}.csv')
        col_schema_strs = []
        for col in columns:
           
            if type(col) == str:
                schema_row = schema_df[schema_df['original_column_name']==col]
            else:
                schema_row = schema_df[schema_df['original_column_name'] == col[0]]
            
            
            if len(schema_row) != 1:
                continue
            
            column_desc = schema_row['column_description'].values[0]
            data_format = schema_row['data_format'].values[0]

            if type(col) == str:
                col_schema_strs.append(f"""COLUMN NAME: {col}
COLUMN DESCRIPTION: {column_desc}
DATA_TYPE: {data_format}                
"""
                )
            
            else:
                col_schema_strs.append(f"""COLUMN NAME: {col[0]}
COLUMN DESCRIPTION: {column_desc}
DATA TYPE: {data_format}
POSSIBLE IMPORTANT VALUE IN COLUMN: {col[1]}
"""
                )
        
        shuffle(col_schema_strs)
        col_schema_str = '\n'.join(col_schema_strs)

        tab_schema_strs.append(f"""{'-'*25}
TABLE NAME: {table_name}

COLUMN INFORMATION: 

{col_schema_str}"""
        )

        shuffle(tab_schema_strs)
        tab_schema_str = '\n'.join(tab_schema_strs)

    database_schema = f"""
DATABASE NAME: {database_name}

SCHEMA: 
{tab_schema_str}
"""
    return database_schema

def divide_and_conquer(
    database_name: str,
    database_path: str,
    ir: list[str],
    question: str,
    hint: str,
    model: LLM
):
    database_schema = get_schema(
        ir = ir,
        database_path= database_path,
        database_name = database_name
    )

    question_str = f"""
QUESTION: {question}
HINT: {hint}
"""
    prompt = f"""{DAC_INSTRUCTION_STR}
***********************
DATABASE_INFO:
{database_schema}    
***********************
{question_str}
***********************
{DAC_ONE_SHOT_EXAMPLE}
***********************
**Divide and Conquer:**
"""

    completion = model(
        response_type = 'single',
        prompts = prompt
    ).result

    completion = completion.split('</think>')[1].strip()
    return completion

def query_plan_cot(
    database_name: str,
    database_path: str,
    ir: list[str],
    question: str,
    hint: str,
    model: LLM
):
    db_schema = get_schema(
        ir = ir, 
        database_name = database_name, 
        database_path = database_path,
    )

    FINAL_PROMPT = f"{QP_PROMPT} \n Now answer the following question using the same technique shown above.\n Database_Info \n *********************** \n **Question**: {question}\n **Evidence**: {hint} **Query Plan**: \n"
    completion = model(
        response_type = 'single',
        prompts = FINAL_PROMPT
    ).result

    completion = completion.split('</think>')[1].strip()
    return completion

def synthetic_examples_gen(
    database_name: str,
    database_path: str,
    ir: list[str],
    model: LLM,
    mode: str = 'icl',
    k: int = 3
):
    assert mode in ['icl', 'regular'], "mode should be one of 'icl', 'regular'"

    db_schema = get_schema(
        ir = ir,
        database_name = database_name,
        database_path = database_path
    )

    if mode == 'regular':
        prompt = SYNTHETIC_EXAMPLE_PROMPT_REG.format(k = k, db_schema = db_schema)

    else: 
        prompt = SYNTHETIC_EXAMPLE_PROMPT_ICL.format(k = k, db_schema = db_schema)

    completion = model(
        response_type = 'single',
        prompts = prompt
    ).result

    return completion

def parse_synthetic_examples(synthetic_examples: list[str]):
    examples = []
    input_pattern = re.compile(r'"input": "(.*?)"')
    output_pattern = re.compile(r'"oputput": "(.*?)"')
    inputs = input_pattern.findall(synthetic_examples)
    outputs = output_pattern.findall(synthetic_examples)
    for inp, out in zip(inputs, outputs):
        examples.append({'input': inp, 'output': out})
    return examples

def run_synth_gen_pipeline(
    question: str,
    database_name: str,
    database_path: str,
    ir: list[str],
    hint: str,
    model: LLM
):

    synthetic_examples = synthetic_examples_gen(
        database_name = database_name,
        database_path = database_path,
        ir = ir,
        mode = 'regular',
        model = model
    )

    synthetic_examples_icl = synthetic_examples_gen(
        database_name = database_name,
        database_path = database_path,
        ir = ir,
        mode = 'icl',
        model = model
    )

    examples = parse_synthetic_examples(synthetic_examples)
    examples.extend(parse_synthetic_examples(synthetic_examples_icl))
    prompt = construct_prompt(
        examples, 
        question,
        database_name,
        database_path,
        ir, 
        hint
    )

    completion = model(
        response_type = 'single',
        prompts = prompt
    ).result

    completion = completion.split('</think>')[1].strip()
    return completion

def construct_prompt(
    examples: list[str],
    question: str,
    database_name: str,
    database_path: str,
    ir: list[str],
    hint: str
):
    db_schema = get_schema(
        ir = ir,
        database_name = database_name,
        database_path = database_path
    )

    prompt = "Use the following examples to generate a SQL query for the given question.\n Output your answer after printing **Final Optimized SQL Query**.\n This will be used to parse.\n\n "
    for example in examples:
        prompt+=f"Example: \nInput: {example['input']}\nOutput: {example['output']}\n\n"

    prompt += f"Database Schema: {db_schema}\n"
    prompt += f"Evidence: {hint} \n"
    prompt += f"Question: {question} \n"
    prompt += "Output:"

    return prompt

def parse_sql_cand(text: str):
    final_sql_pattern = re.compile(
        r'(?i)(?:\*\*final optimized sql query:?\*\*|\*\*final sql query:?\*\*|\*\*optimized sql query:?\*\*|final optimized sql query:?|final sql query:?|final query:?|optimized sql query:?)\s*((SELECT|WITH)[\s\S*?]*?)(?=;|$)',
        re.IGNORECASE
    )
    
    generic_sql_pattern = re.compile(
        r'\b(SELECT|WITH)\b[\s\S]*?(?=;|$)',
        re.IGNORECASE
    )

    final_sql_matches = final_sql_pattern.findall(text)
    if final_sql_matches:
        sql_queries = [match[0].strip() for match in final_sql_matches]
        if sql_queries:
            return sql_queries[-1]
    
    generic_sql_matches = re.findall(generic_sql_pattern, text)
    if generic_sql_matches:
        sql_queries = [match.group().strip() for match in re.finditer(generic_sql_pattern, text)]
        if sql_queries:
            return sql_queries[-1]
    
    return None
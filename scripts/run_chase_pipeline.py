import os
import json
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
import pandas as pd
from openai import OpenAI
import httpx
from time import time
from tqdm import tqdm
import sqlite3
import nltk

from src.info_retrieval.prompts import KEYWORD_FEWSHOT_EXAMPLES
from src.info_retrieval.utils import extract_keyword, semantic_rerank
from src.info_retrieval.lsh import *
from src.cand_gen.utils import divide_and_conquer, parse_sql_cand, query_plan_cot, run_synth_gen_pipeline
from src.model.inference_endpoints import LLM
from src.model.embedding import Embedding
from src.query_fix.utils import query_fixer
from src.selection_agent.utils import select_agent
load_dotenv()

def filter_dict_by_values(data, values_to_find):
    temp = data
    values_to_find_set = set(values_to_find)
    for table_name, columns in temp.items():
        for column_name, column_values in columns.items():
            columns[column_name] = [value for value in column_values if value in values_to_find_set]

    return temp

def info_retrieval(
    question: str,
    hint: str,
    fewshot_examples: list[str],
    lsh_data_path: str,
    model: str,
    embed_model: str,
    http_client: httpx.Client, 
):
    client = OpenAI(
        base_url=os.environ['BASE_URL_MIXTRAL'],
        http_client=http_client,
        api_key=os.environ['API_KEY_MIXTRAL']
    )

    llm = LLM(
        client = client,
        model = model, 
        gen_params = {
            'STREAM': False,
            'TEMPERATURE': 0
        }
    ) # Need to change this function

    kw = extract_keyword(
        question = question,
        hint = hint,
        few_shot_examples = fewshot_examples,
        model=llm
    )

    lsh, minhashes = load_db_lsh(
        db_directory_path = lsh_data_path
    )

    similar_values = []
    for keyword in kw:
        similar_values.append(
            query_lsh(
                lsh=lsh,
                minhashes=minhashes,
                keyword=keyword
            )
        )

    all_values = []
    for similar_value in similar_values:
        for k, values in similar_value.items():
            all_values.extend(values.values())

    all_values = [item for sublist in all_values for item in sublist]
    all_values = list(set(all_values))

    embed_obj = Embedding() # Need to change this function

    semantic_values = []
    for keyword in kw: 
        semantic_values.append(
            semantic_rerank(
                embed_obj = embed_obj,
                strings = all_values,
                keyword = keyword
            )
        )

    filtered_similar_values = []
    for similars, semantics in zip(similar_values, semantic_values):
        filtered_similar_values.append(
            filter_dict_by_values(
                data=similars,
                values_to_find=semantics
            )
        )

    for keyword, fsv in zip(kw, filtered_similar_values):
        for table_name, v in fsv.items():
            for col_name, value_list in v.items():
                edit_dists = []
                for value in value_list:
                    edit_dists.append(nltk.edit_distance(value, keyword))
                if len(edit_dists) > 0:
                    v[col_name] = [value_list[edit_dists.index(sorted(edit_dists)[0])]]

    ir = []
    for filtered_values in filtered_similar_values: 
        for tab, tab_dict in filtered_values.items():
            for col, values in tab_dict.items():
                if len(values) == 0:
                    ir.append(f"`{tab}`.`{col}`")
                else:
                    ir.append(f"`{tab}`.`{col}`.`{values[0]}`")

    return ir

def candidate_generation(
    database_name: str,
    database_path: str,
    ir: list[str],
    question: str,
    hint: str,
    model: str,
    temperature_values: list[float],
    n_schema_shuffles: int,
    http_client: httpx.Client      
):
    client = OpenAI(
        base_url=os.environ['BASE_URL_MIXTRAL'],
        http_client=http_client,
        api_key=os.environ['API_KEY_MIXTRAL']
    )

    all_candidates = []
    log_values = defaultdict(list)

    for temperature in temperature_values:
        llm = LLM(
            client=client,
            model = model,
            gen_params = {
                'STREAM': False,
                'TEMPERATURE': temperature,
                'MAX_NEW_TOKENS': 2048
            }
        )
        for _ in range(n_schema_shuffles):
            dac_answer = divide_and_conquer(
                database_name=database_name,
                database_path=database_path,
                ir=ir,
                question = question,
                hint=hint,
                model=llm
            )

            log_values['DAC_CANDIDATES'].append(dac_answer)

            qp_answer = query_plan_cot(
                database_name=database_name,
                database_path=database_path,
                ir=ir,
                question = question,
                hint=hint,
                model=llm
            )
            log_values['QP_CANDIDATES'].append(qp_answer)

            synth_answer = run_synth_gen_pipeline(
                database_name=database_name,
                database_path=database_path,
                ir=ir,
                question = question,
                hint=hint,
                model=llm               
            )
            log_values['SYNTH_CANDIDATES'].append(synth_answer)

            parsed_dac = parse_sql_cand(dac_answer)
            parsed_qp = parse_sql_cand(qp_answer)
            parsed_synth = parse_sql_cand(synth_answer)
            all_candidates+=[parsed_dac, parsed_qp, parsed_synth]

    return all_candidates, log_values

def query_fix(
    database_name: str,
    database_root_path: str,
    database_path: str,
    candidates: list[str],
    model: str,
    http_client: httpx.Client,
    ir: list[str],
    question: str,
    hint: str,
    n_retries: int = 10
):

    client = OpenAI(
        base_url=os.environ['BASE_URL_DEEPSEEK'],
        http_client=http_client,
        api_key=os.environ['API_KEY_DEEPSEEK']
    )

    llm = LLM(
        client=client,
        model = model,
        gen_params = {
            'STREAM': False,
            'TEMPERATURE': 0.5,
            'MAX_NEW_TOKENS': 2048
        }
    )

    fixed_queries = []
    attempt_counts = []

    all_qr = {}

    for query in candidates:
        intermediate_qr = []
        attempts = 0
        while attempts < n_retries:
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                cursor.execute(query)
                result = cursor.fetchall()
                intermediate_qr.append((query, result))
                conn.close()
                fixed_queries.append((query, result))
                attempt_counts.append(attempts)
                break

            except Exception as e:
                query = query_fixer(
                    database_name=database_name,
                    database_root_path = database_root_path,
                    ir = ir,
                    query_to_correct = query,
                    question = question,
                    hint = hint,
                    result = e,
                    model = llm
                )
                attempts += 1
                intermediate_qr.append((query, e))

    log_values = {"ATTEMPTS": attempt_counts, "INTERMEDIATE_QR": all_qr}
    return fixed_queries, log_values

def select_best_candidate(
    fixed_queries: list[tuple[str, list[tuple[str]]]],
    ir: list[str],
    database_name: str,
    database_path: str,
    question: str,
    hint: str,
    model: str,
    http_client: httpx.Client
):
    client = OpenAI(
        base_url=os.environ['BASE_URL_DEEPSEEK'],
        http_client=http_client,
        api_key=os.environ['API_KEY_DEEPSEEK']
        )

    llm = LLM(
        client = client,
        model = model, 
        gen_params = {
            'STREAM': False,
            'TEMPERATURE': 0,
            'MAX_NEW_TOKENS': 1,
            'STOP_TOKENS': ['*', 'Explanation', '--', '```', '#']
        }
    ) # Need to change this function

    queries = []
    results = []
    for i in fixed_queries:
        queries.append(i[0])
        results.append(i[1])

    best_query = None
    best_result = None

    scores = {query: 0 for query, result in fixed_queries}

    for i in range(len(fixed_queries)):
        for j in range(i+1, len(fixed_queries)):
            candidate_a_query, candidate_a_result = fixed_queries[i]
            candidate_b_query, candidate_b_result = fixed_queries[j]

            if set(candidate_a_result) == set(candidate_b_result):
                scores[candidate_a_query] += 1

            else:
                selected_query = select_agent(
                    ir = ir,
                    database_name = database_name,
                    database_path = database_path,
                    question = question,
                    hint = hint,
                    queries = queries,
                    results = results,
                    model = model, 
                )

                if selected_query == "A":
                    scores[candidate_a_query] += 1
                
                else:
                    scores[candidate_b_query] += 1
    
    if len(scores) > 0:

        best_query = max(scores, key = scores.get)
        best_result = next(result for query, result in fixed_queries if query == best_query)
        logs_sa = scores
        return best_query, best_result, logs_sa
    
    else:
        return None, None, None
    
def main():
    http_client = httpx.Client(verify=False)
    dev_file = 'data/sub_sampled_bird_dev_set.json'

    with open(dev_file, 'r') as file:
        dev_set = json.load(file)
    
    logs = pd.DataFrame(
        columns=[
            'question_id',
            'ground_truth',
            'difficulty',
            'database_name',
            'question',
            'hint',
            'information_retrieval_values',
            'DAC_candidates',
            'QP_candidates',
            'Synth_candidates',
            'Attempts_taken_to_fix',
            'Intermediate_queries_and_results_during_fix',
            'Best_candidate',
            'Best_execution_result',
            'Scores_dictionary',
            'Latency(s)'
        ]
    )

    log_count = 0
    for question in tqdm(dev_set, desc='Processing questions...'):
        start = time()
        try:
            print('Starting info retrieval')
            ir = info_retrieval(
                question = question['question'],
                hint=question['evidence'],
                fewshot_examples=KEYWORD_FEWSHOT_EXAMPLES,
                lsh_data_path=f'{os.environ["DATABASE_ROOT_PATH"]}/{question["db_id"]}',
                model='tgi',
                embed_model='gte-large',
                http_client=http_client
            )

            print("Starting candidate generation \n")
            candidates, log_cg = candidate_generation(
                database_name=question['db_id'],
                database_path=f'{os.environ["DATABASE_ROOT_PATH"]}/{question["db_id"]}',
                ir=ir,
                question=question['question'],
                hint=question['evidence'],
                model='tgi',
                temperature_values=[0, 0.2, 0.5],
                n_schema_shuffles=1,
                http_client=http_client
            )

            print('Starting query fix \n')
            fixed_queries, logs_qf = query_fix(
                database_name=question['db_id'],
                database_root_path=f'{os.environ["DATABASE_ROOT_PATH"]}/{question["db_id"]}',
                database_path=f'{os.environ["DATABASE_ROOT_PATH"]}/{question["db_id"]}/{question["db_id"]}.sqlite',
                candidates=candidates,
                model='tgi',
                http_client=http_client,
                ir=ir,
                question=question['question'],
                hint=question['evidence'],
                n_retries=5
            )

            print('Starting selection of best candidate \n')
            best_query, best_result, logs_sa = select_best_candidate(
                fixed_queries=fixed_queries,
                ir=ir,
                database_name=question['db_id'],
                database_path=f'{os.environ["DATABASE_ROOT_PATH"]}/{question["db_id"]}',
                question=question['question'],
                hint=question['evidence'],
                model='tgi',
                http_client=http_client
            )

            latency = time() - start

            logs.loc[len(logs)] = [
                question['question_id'],
                question['SQL'],
                question['difficulty'],
                question['db_id'],
                question['question'],
                question['evidence'],
                ir,
                log_cg['DAC_CANDIDATES'],
                log_cg['QP_CANDIDATES'],
                log_cg['SYNTH_CANDIDATES'],
                logs_qf['ATTEMPTS'],
                logs_qf['INTERMEDIATE_QR'],
                best_query,
                best_result,
                logs_sa,
                latency
            ]

        except Exception as e:
            print(f'Exception {e} occurred, skipping')

        # Checkpointing
        if len(logs)%4 == 0:
            date = '_'.join(str(datetime.now()).split()).replace('.', '_').replace(':', '-') + f'_run_{log_count+1}'
            logs.to_csv(f"{os.environ['LOGS_SAVE_PATH']}/{date}.csv")
            log_count+=1
            
    date = '_'.join(str(datetime.now()).split()).replace('.', '_').replace(':', '-') + f'_run_{log_count+1}'
    logs.to_csv(f"{os.environ['LOGS_SAVE_PATH']}/{date}.csv")
    log_count+=1
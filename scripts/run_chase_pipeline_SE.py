import os
import json
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
import pandas as pd
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm import tqdm
import sqlite3
import nltk

from src.info_retrieval.prompts import KEYWORD_FEWSHOT_EXAMPLES
from src.info_retrieval.utils import extract_keyword, semantic_rerank
from src.info_retrieval.lsh import *
from src.cand_gen.utils import divide_and_conquer, parse_sql_cand, query_plan_cot, run_synth_gen_pipeline
from src.model.inference_endpoints import LLM
from src.model.embedding import Embedding
from src.query_fix.utils import *
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
):
    client = OpenAI(
        base_url=os.environ['BASE_URL_MIXTRAL'],
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
):
    client = OpenAI(
        base_url=os.environ['BASE_URL_MIXTRAL'],
        api_key=os.environ['API_KEY_MIXTRAL']
    )

    all_candidates = {}
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
            all_candidates[parsed_dac] = 'DAC'
            all_candidates[parsed_qp] = 'QP'
            all_candidates[parsed_synth] = 'SYNTH'

    return all_candidates, log_values

def fetch_results_with_timeout(database_path, query):
    """Function to fetch results with a timeout."""
    # Create a new connection and cursor in each thread
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(query)
    
    # Extract column names only if it's a SELECT query
    if cursor.description is not None:
        columns = [description[0] for description in cursor.description]
    else:
        columns = []  # No columns for non-SELECT queries
    
    result = cursor.fetchall()
    conn.close()  # Close the connection after fetching results
    return columns, result
        
def query_fix(
    database_name: str,
    database_root_path: str,
    database_path: str,
    candidates: dict[str, str],
    model: str,
    ir: list[str],
    question: str,
    hint: str,
    ground_truth: str,
    markup_processor: AutoProcessor,
    markup_model: MarkupLMModel,
    n_retries: int=10,
):
    client = OpenAI(
        base_url=os.environ['BASE_URL_DEEPSEEK'],
        api_key=os.environ['API_KEY_DEEPSEEK']
    )

    llm = LLM(
        client = client,
        model = model, 
        gen_params = {
            'STREAM': False,
            'TEMPERATURE': 0,
            'MAX_NEW_TOKENS': 2048 
        }
    )

    methods = []
    candidates_tmp = []
    for k, v in candidates.items():
        methods.append(v)
        candidates_tmp.append(k)
    
    candidates = candidates_tmp

    fixed_flags = defaultdict(bool)
    fixed_queries = []
    qents = []
    method_percents = []
    all_qr = {}
    attempts = 0

    while attempts < n_retries:
        correct_index = False
        new_candidates = []
        intermediate_qr = []

        for i, query in enumerate(candidates):
            try:
                if fixed_flags[i] == 1:
                    new_candidates.append(query)

                    # Use ThreadPoolExecutor to apply timeout for fetchall
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(fetch_results_with_timeout, database_path, query)
                        try:
                            result, columns = future.result(timeout=15)  # Timeout in seconds

                        except TimeoutError:
                            print(f"Query {i} timed out.")
                            result = []  
                            columns = []

                    intermediate_qr.append((query, result))
                    continue
                else:
                    conn = sqlite3.connect(database_path)
                    cursor = conn.cursor()

                    # Use ThreadPoolExecutor to apply timeout for fetchall
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(fetch_results_with_timeout, database_path, query)
                        try:
                            result, columns = future.result(timeout=5)  # Timeout in seconds
                        except TimeoutError:
                            print(f"Query {i} timed out.")
                            result = []
                            columns = []
                    
                    print(f"query {i}, result {result}")
                    correct_flag = check_exec_accuracy(database_path=database_path, query=query, ground_truth_query=ground_truth)
                    if correct_flag:
                        correct_index = i
                    conn.close()
                    fixed_queries.append((query, result))
                    intermediate_qr.append((query, result))
                    new_candidates.append(query)
                    fixed_flags[i] = 1
            
            except Exception as e:
                fixed_flags[i] = 0
                query = query_fixer(
                    database_name=database_name,
                    database_root_path=database_root_path,
                    ir=ir,
                    query_to_correct=query,
                    question=question,
                    hint=hint,
                    result=e,
                    model=llm
                )
                query = parse_query_fix_output(query)
                new_candidates.append(query)
                intermediate_qr.append((query, e))

        all_qr[attempts] = intermediate_qr
        all_features = []
        
        for cand in new_candidates:
            try:
                # Use ThreadPoolExecutor to apply timeout for fetchall
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(fetch_results_with_timeout, database_path, cand)
                    try:
                        results, columns = future.result(timeout=10)  # Timeout in seconds
                    except TimeoutError:
                        print(f"Query for candidate {cand} timed out.")
                        results = [] 
                        columns = []

                html_result = sql_result_to_html(column_names=columns, result=results)
                features = html_to_features(
                    html_string=html_result, 
                    markup_lm_processor=markup_processor, 
                    markup_lm_model=markup_model,
                )
                
                all_features.append(features.detach().squeeze(dim=0))

            
            except sqlite3.Error as e:
                print(f"An error occurred: {e}")
                html_result = sql_result_to_html(error=e)
                features = html_to_features(
                    html_string=html_result, 
                    markup_lm_processor=markup_processor, 
                    markup_lm_model=markup_model,
                )
                all_features.append(features.detach().squeeze(dim=0))
        
        pi_correct = False
        if correct_index:
            clusters_DB, pi_correct = cluster_sql_queries(embeddings=np.array(all_features), correct_ind=correct_index)
        else:
            clusters_DB = cluster_sql_queries(embeddings=np.array(all_features))
        
        entropy, cluster_percentages = calculate_semantic_entropy(clusters=clusters_DB, methods=methods)
        qents.append(entropy)
        method_percents.append(cluster_percentages)
        attempts += 1

    log_values = {"INTERMEDIATE_QR": all_qr} 
    if pi_correct:
        return fixed_queries, qents, log_values, pi_correct, method_percents
    else:
        return fixed_queries, qents, log_values, None, method_percents


def select_best_candidate(
    fixed_queries: list[tuple[str, list[tuple[str]]]],
    ir: list[str],
    database_name: str,
    database_path: str,
    question: str,
    hint: str,
    model: str,
):
    client = OpenAI(
        base_url=os.environ['BASE_URL_DEEPSEEK'],
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
    dev_file = 'data/sub_sampled_bird_dev_set.json'

    markup_processor = AutoProcessor.from_pretrained("microsoft/markuplm-base")
    markup_model = MarkupLMModel.from_pretrained("microsoft/markuplm-base")

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
            'Intermediate_queries_and_results_during_fix',
            'Query entropies during fix',
            'Probability of bucket of correct query',
            'Method-cluster-distribution',
            'Best_candidate',
            'Best_execution_result',
            'Scores_dictionary',
            'Latency(s)'
        ]
    )

    log_count = 0
    for question in tqdm(dev_set[66:], desc='Processing questions...'):
        start = time.time()
        try:
            print('Starting info retrieval')
            ir = info_retrieval(
                question = question['question'],
                hint=question['evidence'],
                fewshot_examples=KEYWORD_FEWSHOT_EXAMPLES,
                lsh_data_path=f'{os.environ["DATABASE_ROOT_PATH"]}/{question["db_id"]}',
                model='tgi',

            )

            print("Starting candidate generation \n")
            candidates, log_cg = candidate_generation(
                database_name=question['db_id'],
                database_path=f'{os.environ["DATABASE_ROOT_PATH"]}/{question["db_id"]}',
                ir=ir,
                question=question['question'],
                hint=question['evidence'],
                model='tgi',
                temperature_values=[0, 0.2, 0.5, 1.5],
                n_schema_shuffles=1,
            )

            print('Starting query fix \n')
            fixed_queries, qents, logs_qf, pi_correct, method_percents = query_fix(
                database_name=question['db_id'],
                database_root_path=f'{os.environ["DATABASE_ROOT_PATH"]}/{question["db_id"]}',
                database_path=f'{os.environ["DATABASE_ROOT_PATH"]}/{question["db_id"]}/{question["db_id"]}.sqlite',
                candidates=candidates,
                model='tgi',
                ir=ir,
                question=question['question'],
                hint=question['evidence'],
                ground_truth=question['SQL'],
                markup_model=markup_model,
                markup_processor=markup_processor,
                n_retries=8
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
            )

            latency = time.time() - start

            print('LOG LENGTH', len(logs))
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
                logs_qf['INTERMEDIATE_QR'],
                qents,
                pi_correct,
                method_percents,
                best_query,
                best_result,
                logs_sa,
                latency
            ]

        except Exception as e:
            print(f'Exception {e} occurred, skipping')
            time.sleep(10)


        # Checkpointing
        if len(logs)%1 == 0:
            date = '_'.join(str(datetime.now()).split()).replace('.', '_').replace(':', '-') + f'_run_{log_count+1}'
            logs.to_csv(f"{os.environ['LOGS_SAVE_PATH']}/{date}.csv")
            log_count+=1
            time.sleep(10)
            
    date = '_'.join(str(datetime.now()).split()).replace('.', '_').replace(':', '-') + f'_run_{log_count+1}'
    logs.to_csv(f"{os.environ['LOGS_SAVE_PATH']}/{date}.csv")
    log_count+=1
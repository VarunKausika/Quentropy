import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from transformers import AutoProcessor, MarkupLMModel
from src.cand_gen.utils import get_schema
from .prompts import *
from src.model.inference_endpoints import LLM
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import torch

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

def check_exec_accuracy(
    database_path: str,
    query: str, 
    ground_truth_query: str, 
):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
                
    # Execute the query
    cursor.execute(ground_truth_query)
    gt_results = cursor.fetchall()
    gt_columns = [description[0] for description in cursor.description]    
    cursor.execute(query)
    query_results = cursor.fetchall()
    query_columns = [description[0] for description in cursor.description]    
    
    if gt_results == query_results and gt_columns == query_columns:
        return 1

    else:
        return 0

def sql_result_to_html(
    column_names: list[str] = None,
    result: list[tuple[str]] = None,
    error: str = None
):
    

    # Generate HTML content
    html_content = """
<body>
    <table>
        <thead>
            <tr>
"""

    if error:
        html_content += f"                <th>{error}</th>\n"
        html_content += """
    </tr>
</thead>"""

    else:    
        # Add headers to the table
        for header in column_names:
            html_content += f"                <th>{header}</th>\n"

        html_content += """
            </tr>
        </thead>
        <tbody>
"""

        # Add rows for the data
        for row in result:
            html_content += "            <tr>\n"
            for cell in row:
                html_content += f"                <td>{cell}</td>\n"
            html_content += "            </tr>\n"

        # Close the HTML table and body
        html_content += """
        </tbody>
    </table>
</body>
"""
    return html_content

def html_to_features(
    html_string: str,
    markup_lm_processor: AutoProcessor,
    markup_lm_model: MarkupLMModel,
    timeout: int = 30  # Timeout in seconds
):
    try:
        # Define the model inference in a separate thread
        def run_model():
            encoding = markup_lm_processor(html_string, return_tensors="pt", truncation=True)
            outputs = markup_lm_model(**encoding)
            return outputs.last_hidden_state.mean(dim=1)

        # Run the model with a timeout
        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_model)
            last_hidden_states = future.result(timeout=timeout)  # Enforce timeout
            return last_hidden_states

    except TimeoutError:
        print("Model inference timed out, returning random tensor")
        return torch.rand(1, markup_lm_model.config.hidden_size)
    
    except Exception as e:
        print(f"An error occurred during markup processing: {e}, returning random tensor")
        return torch.rand(1, markup_lm_model.config.hidden_size)

# clustering all_features
def cluster_sql_queries(embeddings: np.ndarray, correct_ind: int=None):
    """
    Cluster SQL query embeddings using DBSCAN.
    correct index: tracking variable to track where the correct query ends up
    """
    # Normalize the embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=1)  # You may need to tune these parameters
    clusters = dbscan.fit_predict(normalized_embeddings)
    if correct_ind:
        correct_query_cluster = clusters[correct_ind] 
        pi_correct_cluster = sum([1 if i == correct_query_cluster else 0 for i in clusters])/len(clusters)
        
        return clusters.tolist(), pi_correct_cluster
    else:
        return clusters.tolist()

def calculate_semantic_entropy(clusters: list[int], methods: list[str]) -> float:
    """
    Calculate the semantic entropy of the clusters using the formula: 
    -Sigma(Pi * log(Pi)), where Pi is the number of candidates in cluster i divided by all candidates.
    """
    cluster_counts = Counter(clusters)
    entropy = 0
    for cluster_id, count in cluster_counts.items():
        Pi = count / len(clusters)
        entropy -= Pi * np.log2(Pi)

    cluster_method_counts = defaultdict(lambda: defaultdict(int))

    # Populate the method counts for each cluster
    for cluster, method in zip(clusters, methods):
        cluster_method_counts[cluster][method] += 1

    cluster_percentages = {}
    for cluster, method_counts in cluster_method_counts.items():
        total_count = sum(method_counts.values())
        cluster_percentages[cluster] = {
            method: (count / total_count) for method, count in method_counts.items()
        }
    
    return entropy, cluster_percentages
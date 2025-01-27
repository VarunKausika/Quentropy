import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from transformers import AutoProcessor, MarkupLMModel
from src.cand_gen.utils import get_schema
from .prompts import *
from src.model.inference_endpoints import LLM
import sqlite3

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
    markup_lm_model: MarkupLMModel
):
    # Have to compromise with truncation, max context length of markup lm is 512
    encoding = markup_lm_processor(html_string, return_tensors="pt", truncation=True)

    outputs = markup_lm_model(**encoding)
    last_hidden_states = outputs.last_hidden_state.mean(dim=1)
    return last_hidden_states

# clustering all_features
def cluster_sql_queries(embeddings: np.ndarray, correct_ind: int):
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
    correct_query_cluster = clusters[correct_ind] 
    pi_cluster = sum([1 if i == correct_query_cluster else 0 for i in clusters])
    
    return clusters.tolist()

def calculate_semantic_entropy(clusters: list[int]) -> float:
    """
    Calculate the semantic entropy of the clusters using the formula: 
    -Sigma(Pi * log(Pi)), where Pi is the number of candidates in cluster i divided by all candidates.
    """
    cluster_counts = Counter(clusters)
    entropy = 0
    for cluster_id, count in cluster_counts.items():
        Pi = count / len(clusters)
        entropy -= Pi * np.log2(Pi)
    
    return entropy
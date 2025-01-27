import sqlite3
import os
import random


def get_schema(db_path):
    """Extract schema information from the SQLite database."""
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            # Fetch column information
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [(col[1], col[2]) for col in cursor.fetchall()]  # (column_name, data_type)

            # Fetch foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            foreign_keys = cursor.fetchall()

            schema[table] = {
                "columns": columns,
                "foreign_keys": foreign_keys
            }

        conn.close()
        return schema

    except sqlite3.Error as e:
        print(f"Error accessing database: {e}")
        return None


def save_queries_to_file(case_name, queries, output_dir):
    """Save generated queries to a text file."""
    file_path = os.path.join(output_dir, f"{case_name.replace(' ', '_').lower()}.txt")
    with open(file_path, "w") as file:
        for query in queries:
            file.write(query + "\n")
    print(f"Saved {len(queries)} queries for '{case_name}' in {file_path}")


def generate_simple_queries(tables, schema):
    queries = []
    for table in tables:
        queries.append(f"SELECT * FROM {table};")
        for column, _ in schema[table]["columns"]:
            queries.append(f"SELECT {column} FROM {table};")
    return random.sample(queries, min(len(queries), 50))


def generate_conditional_queries(tables, schema):
    queries = []
    for table in tables:
        for column, col_type in schema[table]["columns"]:
            if "INT" in col_type.upper() or "REAL" in col_type.upper():
                queries.append(f"SELECT * FROM {table} WHERE {column} > 10;")
            queries.append(f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL;")
    return random.sample(queries, min(len(queries), 50))


def generate_join_queries(tables, schema):
    queries = []
    for table, details in schema.items():
        for fk in details["foreign_keys"]:
            foreign_table = fk[2]
            local_column = fk[3]
            foreign_column = fk[4]
            queries.append(
                f"SELECT * FROM {table} JOIN {foreign_table} ON {table}.{local_column} = {foreign_table}.{foreign_column};"
            )
    return random.sample(queries, min(len(queries), 50))


def generate_aggregate_queries(tables, schema):
    queries = []
    for table in tables:
        for column, col_type in schema[table]["columns"]:
            if "INT" in col_type.upper() or "REAL" in col_type.upper():
                queries.append(f"SELECT AVG({column}) FROM {table};")
                queries.append(f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column};")
    return random.sample(queries, min(len(queries), 50))


def generate_nested_queries(tables, schema):
    queries = []
    for table in tables:
        for column, _ in schema[table]["columns"]:
            queries.append(
                f"SELECT * FROM {table} WHERE {column} IN (SELECT {column} FROM {table} WHERE {column} > 10);"
            )
    return random.sample(queries, min(len(queries), 50))


def generate_invalid_queries():
    queries = [
        "SELECT FROM employees;",  # Missing column names
        "SELECT id, name FROM unknown_table;",  # Invalid table
        "SELECT * FROM employees WHERE salary = 'not_a_number';"  # Wrong data type
    ]
    return random.sample(queries * 20, 50)  # Duplicate to make up 50 examples


# New edge cases
def generate_multiple_joins(tables, schema):
    queries = []
    for table in tables:
        for fk1 in schema[table]["foreign_keys"]:
            foreign_table1 = fk1[2]
            local_column1 = fk1[3]
            foreign_column1 = fk1[4]
            for fk2 in schema[foreign_table1]["foreign_keys"]:
                foreign_table2 = fk2[2]
                local_column2 = fk2[3]
                foreign_column2 = fk2[4]
                queries.append(
                    f"SELECT * FROM {table} "
                    f"JOIN {foreign_table1} ON {table}.{local_column1} = {foreign_table1}.{foreign_column1} "
                    f"JOIN {foreign_table2} ON {foreign_table1}.{local_column2} = {foreign_table2}.{foreign_column2};"
                )
    return random.sample(queries, min(len(queries), 50))


def generate_complex_where_conditions(tables, schema):
    queries = []
    for table in tables:
        for column, _ in schema[table]["columns"]:
            queries.append(f"SELECT * FROM {table} WHERE {column} > 100 AND {column} < 500;")
            queries.append(f"SELECT * FROM {table} WHERE {column} = 'SomeValue' OR {column} IS NULL;")
    return random.sample(queries, min(len(queries), 50))


def generate_subqueries_in_select(tables, schema):
    queries = []
    for table in tables:
        for column, _ in schema[table]["columns"]:
            queries.append(
                f"SELECT {column}, (SELECT COUNT(*) FROM {table} WHERE {column} > 50) AS count_subquery FROM {table};"
            )
    return random.sample(queries, min(len(queries), 50))


def generate_subqueries_in_from(tables, schema):
    queries = []
    for table in tables:
        queries.append(
            f"SELECT * FROM (SELECT id, name FROM {table} WHERE name LIKE '%Test%') AS subquery;"
        )
    return random.sample(queries, min(len(queries), 50))


def generate_function_based_queries(tables, schema):
    queries = []
    for table in tables:
        queries.append(f"SELECT UPPER(name) FROM {table};")
        queries.append(f"SELECT AVG(salary) FROM {table} WHERE department = 'Engineering';")
    return random.sample(queries, min(len(queries), 50))


def generate_limit_offset_queries(tables, schema):
    queries = []
    for table in tables:
        queries.append(f"SELECT * FROM {table} LIMIT 10 OFFSET 20;")
        queries.append(f"SELECT id FROM {table} ORDER BY name LIMIT 5;")
    return random.sample(queries, min(len(queries), 50))


def generate_union_queries(tables, schema):
    queries = []
    queries.append(f"SELECT name FROM {tables[0]} UNION SELECT name FROM {tables[1]};")
    queries.append(f"SELECT id FROM {tables[0]} INTERSECT SELECT id FROM {tables[1]};")
    queries.append(f"SELECT name FROM {tables[0]} EXCEPT SELECT name FROM {tables[1]};")
    return random.sample(queries, min(len(queries), 50))


def main():
    # Provide the SQLite database file path
    db_path = f"{os.environ['DATABASE_ROOT_PATH']}/california_schools/california_schools.sqlite"  # Replace with your actual database path

    # Extract schema information
    schema_info = get_schema(db_path)
    if not schema_info:
        print("Failed to retrieve schema. Exiting.")
        return

    # Output directory for query examples
    output_dir = "query_examples_f"
    os.makedirs(output_dir, exist_ok=True)

    # List of tables
    tables = list(schema_info.keys())

    # Generate query examples for each edge case
    examples = {
        "Simple Queries": generate_simple_queries(tables, schema_info),
        "Conditional Queries": generate_conditional_queries(tables, schema_info),
        "Join Queries": generate_join_queries(tables, schema_info),
        "Aggregate Queries": generate_aggregate_queries(tables, schema_info),
        "Nested Queries": generate_nested_queries(tables, schema_info),
        "Invalid Queries": generate_invalid_queries(),
        "Multiple Joins": generate_multiple_joins(tables, schema_info),
        "Complex WHERE Conditions": generate_complex_where_conditions(tables, schema_info),
        "Subqueries in SELECT": generate_subqueries_in_select(tables, schema_info),
        "Subqueries in FROM": generate_subqueries_in_from(tables, schema_info),
        "Function-Based Queries": generate_function_based_queries(tables, schema_info),
        "Limit and Offset Queries": generate_limit_offset_queries(tables, schema_info),
        "Union Queries": generate_union_queries(tables, schema_info),
    }

    # Save examples to files
    for case_name, queries in examples.items():
        save_queries_to_file(case_name, queries, output_dir)

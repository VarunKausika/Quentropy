QP_PROMPT = """
You are an expert data scientist, explain step-by-step using query plan technique how you would generate a sql query answering a question. Here is an example of the query plan technique.
 
 <example>
 Database Info
 {DATABASE_SCHEMA}
 **************************
 Answer Repeating the question and evidence, and generating the SQL with a query plan.
 **Question**: How many Thai restaurants can be found in San Pablo Ave, Albany?
 **Evidence**: Thai restaurant refers to food_type = 'thai'; San Pablo Ave Albany refers to street_name
 = 'san pablo ave' AND T1.city = 'albany'
 **Query Plan**:
 ** Preparation Steps:**
 1. Initialize the process: Start preparing to execute the query.
 2. Prepare storage: Set up storage space (registers) to hold temporary results, initializing them to NULL.
 3. Open the location table: Open the location table so we can read from it.
 4. Open the generalinfo table: Open the generalinfo table so we can read from it.
 ** Matching Restaurants:**
 1. Start reading the location table: Move to the first row in the location table.
 2. Check if the street matches: Look at the street_name column of the current row in location. If it's not
 "san pablo ave," skip this row.
 3. Identify the matching row: Store the identifier (row ID) of this location entry.
 4. Find the corresponding row in generalinfo: Use the row ID from location to directly find the matching
 row in generalinfo.
 5. Check if the food type matches: Look at the food_type column in generalinfo. If it's not "thai," skip
 this row.
 6. Check if the city matches: Look at the city column in generalinfo. If it's not "albany," skip this row.
 ** Counting Restaurants:**
 1. Prepare to count this match: If all checks pass, prepare to include this row in the final count.
 2. Count this match: Increment the count for each row that meets all the criteria.
 3. Move to the next row in location: Go back to the location table and move to the next row, repeating
 the process until all rows are checked.
 4. Finalize the count: Once all rows have been checked, finalize the count of matching rows.
 5. Prepare the result: Copy the final count to prepare it for output.
 ** Delivering the Result:**
 1. Output the result: Output the final count, which is the number of restaurants that match all the
 specified criteria.
 2. End the process: Stop the query execution process.
 3. Setup phase: Before starting the actual query execution, the system prepares the specific values it will
 be looking for, like "san pablo ave," "thai," and "albany."
 **Final Optimized SQL Query:**
 SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2
 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND
 T2.street_name = 'san pablo ave'
 </example>"""

SYNTHETIC_EXAMPLE_PROMPT_REG = """You are a SQLite SQL expert. Your job is to create {k} examples, where each example consists of a
question and a SQL query to fetch the data for it. I want each example to look like this, question input and SQL output pairs:
"input": "What's the description of the series code SM.POP. TOTL for Aruba?
(Hints: Aruba is the name of the country where ShortName = 'Aruba')"
"output": "SELECT T2. Description FROM Country AS T1 INNER JOIN CountryNotes AS T2
ON T1. CountryCode = T2. Countrycode WHERE T1. ShortName = 'Aruba' AND T2. Seriescode =
'SM. POP.TOTL'"
You should generate examples that examine and showcase different aspects and relationshins of the folLowing table schemas, described in "Table creation statements". Understand the database tables and their relationships. Understand the columns and their types and meanings to construct intresting examples.
Generate a mixture of SQL examples that include:
• some simple SQL query examples without JOIN
• some SQL query examples with aggregates, Like COUNT
• some simple SQL query examples with JOIN
• some complex SQL query examples with nested JOIN
***************************
Table creation statements 
{db_schema}
***************************
Generate total of {k} examples. Only outputs the examples (question input and SQL output pairs), and each example can be separated by a new Line"""

SYNTHETIC_EXAMPLE_PROMPT_ICL = """You are a SQLite SQL expert. Your job is to create a set of examples, where each example consists of a
 question and a SQL query to fetch the data for it.
 You should generate examples that examine and showcase different aspects and relationships of the following
 table schemas. Understand the database tables and their relationships. Understand the columns and their
 types and meanings to construct intresting examples.
 I will also show you multiple examples generated for the other database and its table schemas, so you can
 see what kind of examples can be generated for a given database.
 **************************
 ###Examples from other database### The following is the table schemas and column examples for
 other database:
 The database (¨train_db_name¨ ) structure is defined by the following table schemas
 (comments after '-' provide additional column descriptions).
 train_db_schema: ...
 ————————
The folloiwing are the examples generated for the above database schemas:
 Example 1) "input": "Among the countries in the group of Heavily Indebted Poor Countries, how many of
 them are under the lending category of the International Development Associations?
 (Hints: group of Heavily Indebted Poor Countries is OtherGroups = 'HIPC'; International Development
 Associations refers to lendingcategory = 'IDA')"
 "output": "SELECT COUNT(CountryCode) FROM Country WHERE LendingCategory = 'IDA'
 AND OtherGroups = 'HIPC'"
 ...
 Example 10) "input": "What is the description of the footnote on the series code AG.LND.FRST.K2 in
 1990 for Aruba?
 (Hints: Year = 1990; Aruba is the name of country where ShortName = 'Aruba')"
 "output": "SELECT T2.Description FROM Country AS T1 INNER JOIN FootNotes AS T2
 ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = 'Aruba' AND T2.Seriescode =
 'AG.LND.FRST.K2' AND T2.Year = 'YR1990'"
 **************************
 Now similarly, generate examples (question input and SQL output pairs) for the table schemas defined
 below, in "Table creation statements".
 **************************
 ###Table creation statements###
 TARGET_DATABASE_SCHEMA
 **************************
 Only outputs the examples (question input and SQL output pairs), and each example can be separated by
 a new line."""

DAC_INSTRUCTION_STR = """You are a SQL expert, and are tasked with coming up with a sql query that answers a user's question, given a
target database. However, since the user's question may be complex, you may need to come up with multiple sub-questions, generate partial SQL for each sub-question and then aggregate the response into a final query. Provided below are the schema for the target database and an example of how you would decompose and answer a user's question.
DO NOT USE ANY INFORMATION FROM THE EXAMPLE IN YOUR ANSWER. IT IS MEANT ONLY FOR YOUR UNDERSTANDING.
"""

DAC_ONE_SHOT_EXAMPLE = """**1. Divide and Conquer:**
 * **Main Question:** What is the gender of the youngest client who opened account in the low
est average salary branch?
 * **Analysis:** Question asking about 'gender', and it appears in table 'client'. We will use this as the
 output column, selecting it from the youngest client in the lowest average salary branch.
 * **Pseudo SQL:** SELECT 'T1'.'gender' FROM 'client' AS 'T1' WHERE youngest client in the lowest
 average salary branch
 * **Sub-question 1:** youngest client in the lowest average salary branch
 * **Analysis:** According to the hint, we need to use the 'A11' from 'district' to get the salary info,
 and the youngest client can be obtained from using the 'birth_date' column of table 'client'. The items
 between these two tables can be INNER JOIN using district_id.
 * **Pseudo SQL:** SELECT 'T1'.'client_id' FROM 'client' AS 'T1' INNER JOIN 'district' AS 'T2' ON
 'T1'.'district_id' = 'T2'.'district_id' WHERE lowest average salary branch ORDER BY 'T1'.'birth_date'
 DESC NULLS LAST LIMIT 1
 * **Sub-question 1.1:** lowest average salary branch
 * **Analysis:** We can get the lowest average salary branch using order by 'A11' ASC and pick top 1. The
 column 'A11' is not NULLABLE, so we do not need to add "IS NOT NULL" filter * **Pseudo SQL:**
 SELECT 'district_id' FROM 'district' ORDER BY 'A11' ASC LIMIT 1
 **2. Assembling SQL:**
 * **Sub-question 1.1 (lowest average salary branch):** * **SQL:** SELECT 'district_id' FROM
 'district' ORDER BY 'A11' ASC LIMIT 1
 * **Sub-question 1 (youngest client in the lowest average salary branch):**
 * **SQL:** SELECT 'T1'.'client_id' FROM 'client' AS 'T1' INNER JOIN 'district' AS 'T2' ON
 'T1'.'district_id' = 'T2'.'district_id' WHERE 'T2'.'district_id' IN (SELECT 'district_id' FROM 'district'
 ORDER BY 'A11' ASC LIMIT 1) ORDER BY 'T1'.'birth_date' DESC NULLS LAST LIMIT 1
 * **Main Question (gender of the client):**
 * **SQL:** SELECT 'T1'.'gender' FROM 'client' AS 'T1' WHERE 'T1'.'client_id' = (SELECT
 'T1'.'client_id' FROM 'client' AS 'T1' INNER JOIN 'district' AS 'T2' ON 'T1'.'district_id' =
 'T2'.'district_id' WHERE 'T2'.'district_id' IN (SELECT 'district_id' FROM 'district' ORDER BY 'A11'
 ASC LIMIT 1) ORDER BY 'T1'.'birth_date' DESC NULLS LAST LIMIT 1)
 **3. Simplification and Optimization:**
 * The nested queries can be combined using a single 'INNER JOIN' and the filtering can be done within a
 single 'ORDER BY' clause.
 **Final Optimized SQL Query:**
 SELECT 'T1'.'gender' FROM 'client' AS 'T1' INNER JOIN 'district' AS 'T2' ON 'T1'.'district_id' =
 'T2'.'district_id' ORDER BY 'T2'.'A11' ASC, 'T1'.'birth_date' DESC NULLS LAST LIMIT 1"""



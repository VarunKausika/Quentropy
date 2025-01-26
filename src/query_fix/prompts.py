QF_EXAMPLES = """### Example 1: Incorrect Column Name
**Original Question:** Retrieve the names of all employees in the 'employees' table.
**Hint:** The column name for employee names is 'employee_name'.
**Executed SQL Query:** SELECT name FROM employees
**Execution Result:** Error: column 'name' does not exist.
**Corrected Query:** SELECT employee_name FROM employees

### Example 2: Missing Join Condition
**Original Question:** List all orders with the corresponding customer names.
**Hint:** The 'orders' table has a 'customer_id' column that references the 'customers' table.
**Executed SQL Query:** SELECT orders.order_id, customers.customer_name FROM orders, customers
**Execution Result:** Returns a Cartesian product of orders and customers.
**Corrected Query:** SELECT orders.order_id, customers.customer_name FROM orders JOIN customers ON orders.customer_id = customers.customer_id

### Example 3: Incorrect Table Alias
**Original Question:** Find the total sales for each product.
**Hint:** The 'sales' table has 'product id' and 'amount' columns.
**Executed SQL Query:** SELECT 'product id', SUM(amount) FROM sales GROUP BY s.product_id
**Execution Result:** Error: column 'product_id' does not exist.
**Corrected Query:** SELECT s.'product id', SUM(s.amount) FROM sales s GROUP BY s.product_id

### Example 4: Using WHERE Instead of HAVING
**Original Question:** Retrieve departments with more than 10 employees.
**Hint:** Use the 'departments' and 'employees' tables.
**Executed SQL Query:** SELECT 'department id', COUNT(*) FROM employees GROUP BY department_id WHERE COUNT(*) > 10
**Execution Result:** Syntax error at or near "WHERE".
**Corrected Query:** SELECT 'department id', COUNT(*) FROM employees GROUP BY department_id HAVING COUNT(*) > 10

### Example 5: Incorrect Aggregate Function
**Original Question:** Find the average salary of employees in each department.
**Hint:** The 'employees' table has 'department_id' and 'salary' columns.
**Executed SQL Query:** SELECT department_id, SUM(salary) FROM employees GROUP BY department_id
**Execution Result:** Returns the sum of salaries instead of the average.
**Corrected Query:** SELECT department_id, AVG(salary) FROM employees GROUP BY department_id

### Example 6: Incorrect Use of DISTINCT
**Original Question:** Retrieve unique job titles from the 'employees' table.
**Hint:** Use the 'job_title' column.
**Executed SQL Query:** SELECT DISTINCT(job_title) FROM employees
**Execution Result:** Error: syntax error at or near "DISTINCT".
**Corrected Query:** SELECT DISTINCT job_title FROM employees

### Example 7: Incorrect Subquery Placement
**Original Question:** Find employees who earn more than the average salary.
**Hint:** Use a subquery to calculate the average salary.
**Executed SQL Query:** SELECT 'employee name' FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)
**Execution Result:** Errors: subquery returned more than one row.
**Corrected Query:** SELECT 'employee name' FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)

### Example 8: Using COUNT() Incorrectly
**Original Question:** Count the number of employees in each department.
**Hint:** Use the 'department_id' column.
**Executed SQL Query:** SELECT department_id, COUNT(*) FROM employees GROUP BY department_id, COUNT(department_id)
**Execution Result:** Syntax error.
**Corrected Query:** SELECT department_id, COUNT(*) FROM employees GROUP BY department_id

### Example 9: Incorrect Date Format
**Original Question:** Retrieve orders placed on '2023-01-01'.  
**Hint:** Use the 'order date' column.  
**Executed SQL Query:** `SELECT * FROM orders WHERE order_date = '01-01-2023';`  
**Execution Result:** Error: invalid date format.  
**Corrected Query:** Final Answer: `SELECT * FROM orders WHERE order_date = '2023-01-01';`

### Example 10: Missing GROUP BY Clause
**Original Question:** Find the total sales for each product category.  
**Hint:** Use the 'sales' and 'products' tables.  
**Executed SQL Query:** `SELECT product_category, SUM(amount) FROM sales JOIN products ON sales.product_id = products.product_id;`  
**Execution Result:** Error: column "product_category" must appear in the GROUP BY clause.  
**Corrected Query:** Final Answer: `SELECT product_category, SUM(amount) FROM sales JOIN products ON sales.product_id = products.product_id GROUP BY product_category;`"""

QF_PROMPT = """**Task Description:**
 You are an SQL database expert tasked with correcting a SQL query. A previous attempt to run a query
 did not yield the correct results, either due to errors in execution or because the result returned was empty
 or unexpected. Your role is to analyze the error based on the provided database schema and the details of
 the failed execution, and then provide a corrected version of the SQL query.
 **Procedure:**
 1. Review Database Schema:- Examine the table creation statements to understand the database structure.
 2. Analyze Query Requirements:- Original Question: Consider what information the query is supposed to retrieve.- Hint: Use the provided hints to understand the relationships and conditions relevant to the query.- Executed SQL Query: Review the SQL query that was previously executed and led to an error or
 incorrect result.- Execution Result: Analyze the outcome of the executed query to identify why it failed (e.g., syntax
 errors, incorrect column references, logical mistakes).
 3. Correct the Query:- Modify the SQL query to address the identified issues, ensuring it correctly fetches the requested data
 according to the database schema and query requirements.
 **Output Format:**
 Present your corrected query as a single line of SQL code, after Final Answer. Ensure there are
 no line breaks within the query.
 Here are some examples:
 {EXAMPLES}
 ======= Your task =======
 **************************
 Table creation statements
 {DATABASE_SCHEMA}
 **************************
 The original question is:
 Question:
 {QUESTION}
 Evidence:
 {HINT}
 The SQL query executed was:
 {QUERY}
 The execution result:
 {RESULT}
 **************************
 Based on the question, table schema and the previous query, analyze the result try to fix the query. Never ever provide an explanation. Just end with the correct SQL query.
 Final Answer: 
 ```sql
 """
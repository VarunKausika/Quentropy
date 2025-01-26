SA_PROMPT = """Instruction:
 Given the DB info and question, there are two candidate queries. There is correct one and incorrect one,
 compare the two candidate answers, analyze the differences of the query and the result. Based on the
 original question and the provided database info, choose the correct one.
 **************************
 Database Schema
 {DATABASE_SCHEMA}
 **************************
 Question:
 {QUESTION}
 Evidence:
 {HINT}
 **************************
 Candidate A
 {CANDIDATE_A_QUERY}
 Execution result
 {CANDIDATE_A_RESULT}
 **************************
 Candidate B
 {CANDIDATE_B_QUERY}
 Execution result
 {CANDIDATE_B_RESULT}
 Just output the correct answer "A" or "B". Never ever provide an explanation.
 Answer:
 """
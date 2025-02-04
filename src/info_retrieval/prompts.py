KEYWORD_FEWSHOT_EXAMPLES= """Question: Whats the fastest lap time ever in a race for Lewis Hamilton?
["FASTEST LAP TIME", "LEWIS HAMILTON"]"""

EXTRACT_KEYWORDS = """Objective: Analyze the given question and hint to identify and extract
keywords, keyphrases, and named entities. These elements are crucial for
understanding the core components of the inquiry and the guidance provided.
This process involves recognizing and isolating significant terms and phrases
that could be instrumental in formulating searches or queries related to the
posed question.
Instructions:

Read the Question Carefully: Understand the primary focus and specific
details of the question. Look for any named entities (such as organizations,
locations, etc.), technical terms, and other phrases that encapsulate important
aspects of the inquiry.
Analyze the Hint: The hint is designed to direct attention toward certain
elements relevant to answering the question. Extract any keywords, phrases, or
named entities that could provide further clarity or direction in formulating
an answer.
List Keyphrases and Entities: Combine your findings from both the question
and the hint into a single Python list. This list should contain:


Keywords: Single words that capture essential aspects of the question or
hint.
Keyphrases: Short phrases or named entities that represent specific concepts,
locations, organizations, or other significant details.
Ensure to maintain the original phrasing or terminology used in the question
and hint.

{few_shot_examples}
Task:
Given the following question and hint, identify and list all relevant keywords,
keyphrases, and named entities.
Question: {question}
Hint: {hint}
Please capture the essence of both the question and hint through the identified terms and phrases.
Only output the Python List, no explanations needed. Do not include reasoning in the final answer. Just output the list.
Please strictly follow only the below provided answer format
ANSWER FORMAT:
['keyword_1', 'keyword_2', 'keyword_3', ...]
ANSWER:
"""
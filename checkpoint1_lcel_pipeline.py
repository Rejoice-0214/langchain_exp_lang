from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import re

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

parser = StrOutputParser()

# -------------------------
# 1. DECOMPOSER
# -------------------------

decompose_prompt = ChatPromptTemplate.from_template("""
You are a reasoning assistant.

Decompose the question below into at most 3 numbered sub-questions.
Only output a numbered list.

Question:
{question}
""")

decomposer = decompose_prompt | llm | parser


def parse_numbered_list(text: str):
    lines = text.strip().split("\n")
    subs = []
    for line in lines:
        match = re.match(r"\s*\d+\.\s*(.*)", line)
        if match:
            subs.append(match.group(1))
    return subs[:3]


decomposer_chain = decomposer | RunnableLambda(parse_numbered_list)

# -------------------------
# 2. SUB-QUESTION ANSWERER
# -------------------------

answer_prompt = ChatPromptTemplate.from_template("""
Answer the sub-question below.

Return plain text in this format:

Answer: <one short paragraph>
Steps:
- step 1
- step 2
- step 3

Sub-question:
{subquestion}
""")

answer_chain = answer_prompt | llm | parser

# -------------------------
# 3. COMBINER
# -------------------------

combine_prompt = ChatPromptTemplate.from_template("""
You are given:
- the original question
- sub-questions
- their answers

Write a final response in exactly 3 short lines.

Original question:
{question}

Sub-questions:
{subs}

Sub-answers:
{answers}
""")

combiner = combine_prompt | llm | parser

# -------------------------
# 4. FULL LCEL PIPELINE
# -------------------------

def run_pipeline(question: str):
    subs = decomposer_chain.invoke({"question": question})

    # batch = parallel efficiency
    answers = answer_chain.batch(
        [{"subquestion": s} for s in subs]
    )

    final = combiner.invoke({
        "question": question,
        "subs": "\n".join([f"{i+1}. {s}" for i, s in enumerate(subs)]),
        "answers": "\n\n".join(answers)
    })

    return subs, answers, final


if __name__ == "__main__":
    questions = [
        "How does gradient descent work and why is it important in machine learning?",
        "What are the main differences between SQL and NoSQL databases?"
    ]

    for q in questions:
        print("\n" + "="*80)
        print("QUESTION:", q)

        subs, answers, final = run_pipeline(q)

        print("\n--- Decomposed sub-questions ---")
        for i, s in enumerate(subs, 1):
            print(f"{i}. {s}")

        print("\n--- Sub-answers ---")
        for a in answers:
            print("\n" + a)

        print("\n--- Final synthesis ---")
        print(final)

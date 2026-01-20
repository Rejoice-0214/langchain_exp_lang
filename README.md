This assignment demonstrates LCEL composition by building a multi-stage reasoning pipeline. LCEL pipes (|) were used to connect prompts, models, and output parsers into reusable runnables for decomposition, sub-question answering, and final synthesis. 
The decomposer converts a question into a numbered list of at most three sub-questions, which are parsed using a custom text parser. 
Sub-questions are answered using the batch() method, allowing parallel execution for efficiency instead of sequential LLM calls. Finally, a combiner runnable synthesizes all intermediate outputs into a concise three-line answer. 
Batch processing improves performance and reflects real-world GenAI system design, where independent tasks are executed concurrently to reduce latency and cost.

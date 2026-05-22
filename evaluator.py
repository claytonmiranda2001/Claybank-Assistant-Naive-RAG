#Imports

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Evaluator

class ExpectedAnswerEvaluator:

    PROMPT_TEMPLATE = """
    You are an evaluator agent for ClayBank Assistant.

    Your task is to compare:

    1. Customer question
    2. Agent response
    3. Expected response

    Determine if the agent response matches the expected response.

    Evaluation Rules:

    Return:
    - 1 = Correct response
    - 0 = Incorrect response

    Consider:
    - Semantic meaning
    - Correctness
    - Relevance
    - Completeness

    The wording does NOT need to be identical.

    IMPORTANT:
    Return ONLY:
    1
    or
    0

    QUESTION:
    {question}

    AGENT RESPONSE:
    {agent_response}

    EXPECTED RESPONSE:
    {expected_response}
    """
    # Model and prompt initialization
    def __init__(self):
        # Evaluator model
        self.model = ChatOllama(
            model="llama3"
        )
        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            self.PROMPT_TEMPLATE
        )
    # Evaluation method
    def evaluate(
        self,
        question,
        agent_response,
        expected_response
    ):
        # Format the prompt with the provided inputs
        prompt = self.prompt_template.format(
            question=question,
            agent_response=agent_response,
            expected_response=expected_response
        )
        # Generating the evaluation
        evaluation = self.model.invoke(prompt)

        result = evaluation.content.strip()

        if result not in ["0", "1"]:
            return 0

        return int(result)
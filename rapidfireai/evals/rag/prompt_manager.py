from copy import deepcopy
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector, SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import hashlib
import json
import asyncio
import concurrent.futures

class PromptManager:
    """
    A manager class for handling prompt templates with few-shot examples.

    This class manages the creation and formatting of prompts that include
    instructions and few-shot examples. It supports loading instructions from
    either a string or a file, and uses an example selector to dynamically
    choose relevant examples for each query based on semantic similarity.

    Attributes:
        instructions (str): The main instructions for the prompt.
        instructions_file_path (str): Path to a file containing instructions.
        examples (List[Dict[str, str]]): List of example dictionaries for few-shot learning.
        embedding_cls (Type[Embeddings]): The embedding class to instantiate.
        embedding_kwargs (Dict[str, Any]): Parameters for initializing the embedding class.
        example_selector_cls (Type[Union[MaxMarginalRelevanceExampleSelector, SemanticSimilarityExampleSelector]]):
            Class for creating example selectors. Must be either MaxMarginalRelevanceExampleSelector
            or SemanticSimilarityExampleSelector.
        example_prompt_template (PromptTemplate): Template for formatting individual examples.
        k (int): Number of examples to retrieve for few-shot learning.
        example_selector (BaseExampleSelector): The instantiated example selector.
        fewshot_generator (FewShotPromptTemplate): The configured few-shot prompt generator.
    """

    def __init__(
        self,
        instructions: str = "",
        instructions_file_path: str = "",
        examples: list[dict[str, str]] = [],
        embedding_cls: type[Embeddings] = None,  # Class like HuggingFaceEmbeddings, OpenAIEmbeddings, etc
        embedding_kwargs: dict[str, Any] | None = None,
        example_selector_cls: type[MaxMarginalRelevanceExampleSelector | SemanticSimilarityExampleSelector] = None,
        example_prompt_template: PromptTemplate = None,
        k: int = 3,
    ) -> None:
        """
        Initialize the PromptManager with instructions and example handling components.

        Args:
            instructions (str): The main instructions text for the prompt. Can be empty
                if instructions_file_path is provided.
            instructions_file_path (str): Path to a file containing instructions. Used
                if instructions is empty or None.
            examples (List[Dict[str, str]]): List of example dictionaries for few-shot learning.
                Each dictionary should contain the example data (e.g., {'input': '...', 'output': '...'}).
            embedding_cls (Type[Embeddings]): The embedding class to instantiate for semantic similarity.
                Examples: HuggingFaceEmbeddings, OpenAIEmbeddings, etc.
            embedding_kwargs (Dict[str, Any]): Parameters for initializing the embedding class.
                Must contain all required parameters for the chosen embedding class.
            example_selector_cls (Type[Union[MaxMarginalRelevanceExampleSelector, SemanticSimilarityExampleSelector]]):
                Class for creating example selectors. Must be either MaxMarginalRelevanceExampleSelector
                or SemanticSimilarityExampleSelector.
            example_prompt_template (PromptTemplate): Template for formatting individual
                examples in the few-shot prompt.
            k (int): Number of examples to retrieve for few-shot learning. Defaults to 3.

        Note:
            Either instructions or instructions_file_path must be provided. All other parameters
            are required for few-shot functionality with semantic similarity.
        """
        self.instructions = instructions
        self.instructions_file_path = instructions_file_path
        self.examples = examples
        self.embedding_cls = embedding_cls
        self.embedding_kwargs = embedding_kwargs
        self.example_selector_cls = example_selector_cls
        self.example_prompt_template = example_prompt_template
        self.k = k

    def setup_examples(self) -> None:
        """
        Set up example selector and few-shot generator.

        This method performs validation and setup:
        - Loads instructions from file if not provided directly
        - Validates that required components are present
        - Creates the FewShotPromptTemplate for generating examples

        Raises:
            ValueError: If required components are missing or if neither instructions
                nor instructions_file_path is provided.
            FileNotFoundError: If instructions_file_path is provided but the file doesn't exist.
        """
        if not self.instructions:
            if not self.instructions_file_path:
                raise ValueError("either instructions or instructions_file_path is required")
            with open(self.instructions_file_path) as f:
                self.instructions = f.read()
        if not self.embedding_cls:
            raise ValueError("embedding_cls is required")
        if not self.embedding_kwargs:
            raise ValueError("embedding_kwargs is required")
        if not self.example_selector_cls:
            raise ValueError("example_selector_cls is required")

        # Validate that the example selector class is one of the supported types
        if self.example_selector_cls not in [
            MaxMarginalRelevanceExampleSelector,
            SemanticSimilarityExampleSelector,
        ]:
            raise ValueError(
                f"example_selector_cls must be either MaxMarginalRelevanceExampleSelector or "
                f"SemanticSimilarityExampleSelector, got: {self.example_selector_cls.__name__}"
            )
        if not self.example_prompt_template:
            raise ValueError("example_prompt_template is required")
        if not self.k:
            raise ValueError("k is required")

        self.embedding_function = self.embedding_cls(**self.embedding_kwargs)

        self.vectorstore_cls = FAISS
        # self.vectorstore_cls_kwargs = {
        #     "index": faiss.IndexFlatL2(len(self.embedding_function.embed_query("RapidFire AI is awesome!"))),
        #     "docstore": InMemoryDocstore(),
        #     "index_to_docstore_id": {},
        # }

        self.example_selector = self.example_selector_cls.from_examples(
            self.examples,
            self.embedding_function,
            vectorstore_cls=self.vectorstore_cls,
            k=self.k,
            # **self.vectorstore_cls_kwargs
        )

        self.fewshot_generator = FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=self.example_prompt_template,
            prefix="",
            suffix="",
            input_variables=["user_query"],
        )

    def get_instructions(self) -> str:
        """
        Get the current instructions text.

        Returns:
            str: The instructions text that will be used in prompts.
        """
        return self.instructions

    def get_fewshot_examples(self, user_queries: list[str]) -> list[str]:
        """
        Generate few-shot examples formatted according to the template.

        Uses the example selector to choose relevant examples based on the user query,
        then formats them using the configured example prompt template.

        Args:
            user_query (str, optional): The user's query to use for example selection.
                Defaults to empty string.

        Returns:
            str: Formatted few-shot examples ready to be included in a prompt.

        Note:
            This method requires that setup_examples() has been called first to set up
            the fewshot_generator.
        """
        async def gather_examples():
            """Async helper to gather all examples concurrently"""
            tasks = [
                self.fewshot_generator.aformat(user_query=user_query)
                for user_query in user_queries
            ]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_running_loop()
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(gather_examples())
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                fewshot_examples = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            fewshot_examples = asyncio.run(gather_examples())

        return fewshot_examples

    def copy(self) -> "PromptManager":
        """
        Create a deep copy of this PromptManager instance.

        Returns a new PromptManager instance with copies of all attributes.
        The copied instance will have the same configuration but can be
        modified independently of the original.

        Returns:
            PromptManager: A new PromptManager instance that is a deep copy
                of the current instance.

        Note:
            The copied instance will need to be set up by calling
            setup_examples() before it can be used to generate examples.
        """
        return PromptManager(
            instructions=self.instructions,
            instructions_file_path=self.instructions_file_path,
            examples=deepcopy(self.examples),
            embedding_cls=self.embedding_cls,
            embedding_kwargs=deepcopy(self.embedding_kwargs),
            example_selector_cls=self.example_selector_cls,
            example_prompt_template=deepcopy(self.example_prompt_template),
            k=self.k,
        )

    def __getstate__(self):
        """
        Custom pickling to exclude unpicklable embedding_function.

        The embedding_function contains threading locks that can't be pickled.
        We'll recreate it when unpickling using embedding_cls and embedding_kwargs.
        """
        state = self.__dict__.copy()
        # Remove unpicklable objects
        state['embedding_function'] = None
        state['example_selector'] = None
        state['fewshot_generator'] = None
        return state

    def __setstate__(self, state):
        """
        Custom unpickling to restore state without embedding_function.

        The embedding_function will be recreated by calling setup_examples()
        after deserialization.
        """
        self.__dict__.update(state)

    def get_hash(self) -> str:
        """
        Generate a unique hash for this prompt manager configuration.

        Used for deduplicating contexts in the database - if two pipelines have
        identical prompt configurations, they can share the same context.

        Returns:
            SHA256 hash string
        """
        prompt_dict = {
            "instructions": self.instructions,
            "k": self.k,  # Number of fewshot examples to retrieve
            "embedding_cls": self.embedding_cls.__name__ if self.embedding_cls else None,
            "embedding_kwargs": self.embedding_kwargs,  # Model name and config
            "example_selector_cls": self.example_selector_cls.__name__ if self.example_selector_cls else None,
            "num_examples": len(self.examples) if self.examples else 0,
            # Hash the examples themselves to detect changes
            "examples_hash": hashlib.sha256(
                json.dumps(self.examples, sort_keys=True).encode()
            ).hexdigest()
            if self.examples
            else None,
        }
        prompt_json = json.dumps(prompt_dict, sort_keys=True)
        return hashlib.sha256(prompt_json.encode()).hexdigest()


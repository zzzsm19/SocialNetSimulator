"""
@Name: recagent_memory.py
@Author: Hao Yang, Zeyu Zhang
@Date: 2023/8/10

Script: This is the memory module for recagent.
"""

import logging
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from random import random

from llm.llm import LLM

logger = logging.getLogger(__name__)


class AgentRetriever():
    """
    RecAgentRetriever is to retrieve memories from long-term memory module based on memory salience, importance and recency.
    """
    def __init__(self):
        self.memory_stream = []
        self.now = None
        self.recency_weight = 0.3
        self.relevance_weight = 0.4
        self.importance_weight = 0.3

    def add_documents(self, documents: List[dict], current_time: Optional[datetime] = None) -> List[dict]:
        """Add documents to the memory stream."""
        if current_time is None:
            current_time = datetime.now()
        for doc in documents:
            doc["created_at"] = current_time
            doc["last_accessed_at"] = current_time
            self.memory_stream.append(doc)

    def _get_combined_score(self, doc: dict, relevance: float, current_time: datetime) -> float:
        """Calculate the combined score of a document."""
        return relevance * self.relevance_weight + \
            (1.0 - (current_time - doc["last_accessed_at"]).total_seconds() / 3600) * self.recency_weight + \
            doc["importance"] * self.importance_weight

    def get_relevant_documents(self, query: str) -> List[dict]:
        """Return documents that are relevant to the query."""
        current_time = self.now
        docs_and_scores = [
            (doc, 0.0)
            # Calculate for all memories.
            for doc in self.memory_stream
        ]
        # If a doc is considered salient, update the salience score
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        # Ensure frequently accessed memories aren't forgotten
        retrieved_num = 0
        for doc, _ in rescored_docs:
            if retrieved_num < self.k and doc['content'].find('[FORGET]') == -1 \
                    and doc['content'].find('[MERGE]') == -1:
                retrieved_num += 1
                buffered_doc = self.memory_stream[doc["buffer_idx"]]
                buffered_doc["last_accessed_at"] = current_time
                result.append(buffered_doc)
        return result


class SensoryMemory():
    """
    Sensory memory is intended to receive the observations (that are ready to be stored as memories) from the environment,
    extract and summarize important elements by attention mechanism, and output them to short term memory.
    """

    def __init__(self, llm, buffer_size=3):
        self.llm = llm
        self.buffer_size = buffer_size
        self.importance_weight = 0.9
        self.buffer = []

    def clear(self):
        self.buffer = []

    def _score_memory_importance(self, observation: str) -> float:
        """
        Obtain the importance score of this memory.
        :param observation: The text of the observation.
        :return: (float) The importance of this observation.
        """
        prompt = "请给下面的感知记忆打分，分数越高表示越重要，分数范围是0到9。注意，打分时应该更多地关注记忆的重要性。\n" \
                    "请回答一个数字。\n"
        prompt += observation
        score = self.llm.invoke(prompt).strip()
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def add_sensory_memory(self, obs):
        """
        This function is only called in the function RecAgentMemory.save_context(). It is used to transport observations to a piece of short term memory.
        For each time, it receives only one observation, and adds into buffer. If buffer is full, then converts them into a piece of term memory.
        :param obs: The observation that is ready to transport to short term memory.
        :return: (1)Buffer full: List of tuple (score[float], stm[str]). (2) Buffer not full: None.
        """
        # Add the observation into the buffer.
        self.buffer.append(obs)

        if len(self.buffer) >= self.buffer_size:
            return self.dump_shortTerm_list()
        else:
            return None

    def dump_shortTerm_list(self):
        """
        Convert all the observations in buffer to a piece of short term memory, and clear the buffer.
        :return: List of tuple (score[float], stm[str])
        """
        
        prompt = "这有一些感知记忆：\n"
        for ind, obs in enumerate(self.buffer):
            prompt += "[%d] %s\n" % (ind, obs)
        prompt += "请将上述感知记忆总结成一句话。如果感知记忆中有人名，请使用第三人称，否则使用第一人称。" \
                    # "注意，总结应该更多地关注。" \
        result = self.llm.invoke(prompt).strip()
        result = [(self._score_memory_importance(result), result)]
        # # Remove the short term memory whose importance score is lower than a threshold.
        # result = [text for text in result if text[0] > 0.62]

        # Clear the buffer.
        self.clear()
        if len(result) != 0:
            return result
        else:
            return None


class ShortTermMemory():
    """
    The short-term memory module is to temporally store the observations from sensory memory module,
    which can be enhanced by other observations or retrieved memories to enter the long-term memory module.
    """
    def __init__(self, llm):
        self.llm = llm
        self.verbose: bool = False
        self.capacity: int = 10
        self.short_memories: List[str] = []
        self.short_embeddings: List[List[float]] = []
        self.memory_importance: List[float] = []
        self.enhance_cnt: List[int] = [0 for _ in range(self.capacity)]
        self.enhance_memories: List[List[str]] = [[] for _ in range(self.capacity)]
        self.enhance_threshold: int = 3

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def add_short_memory(self, observation: str, importance: float, op: str = 'add'):
        """
        Add a new observation into short-term memory, and return the enhanced short-term memory and with the insight.
        :param observation: the content of the sensory memory of retrieved memory
        :param imporatance: the importance score of observation
        :param op: specify the types of observation. "add" means that the observation is sensory memory,
                 "retrieval" means that the observation is the retrieved memory.
        """
        const = 0.1
        # compute the vector similarities between observation and the existing short-term memories
        observation_embedding = self.llm.get_embedding(observation)
        for idx, memory_embedding in enumerate(self.short_embeddings):
            similarity = self.cosine_similarity(observation_embedding, memory_embedding)
            # primacy effect
            # The following one line was corrected by Zeyu on 23.8.27-7pm. Ori: if idx + 1 == len(short_term_embeddings):
            if idx + 1 == len(self.short_embeddings):
                similarity += const
            # sample and select the enhanced short-term memory
            # Sigmoid function
            prob = 1 / (1 + np.exp(-similarity))
            if prob >= 0.7 and random() <= prob:
                self.enhance_cnt[idx] += 1
                self.enhance_memories[idx].append(observation)
        memory_content, memory_importance, insight_content = self.transfer_memories(observation)
        if op == 'add':
            self.short_memories.append(observation)
            self.memory_importance.append(importance)
            self.short_embeddings.append(observation_embedding)
            self.discard_memories()
        return memory_content, memory_importance, insight_content

    def get_short_term_insight(self, content: str):
        """
        Get insight of the short-term memory and other memories or observations that enhance that short-term memory.
        :param content: short-term memory and other memories or observations that enhance that short-term memory
        :return: (List[str]) The insight of the short-term memory.
        """
        prompt = "这有一些短期记忆：\n"
        for ind, cont in enumerate(content.split(';')):
            prompt += "[%d] %s\n" % (ind, cont)
        prompt += "请从上述短期记忆中总结出一个高层次的insight。用一个句子回答。"
        result = self.llm.invoke(prompt).strip()
        return self._parse_list(result)

    def transfer_memories(self, observation):
        """
        Transfer all possible short-term memories to long-term memory.
        :param observation: the observation enters the short-term memory or the retrieved memory
        :return
            (List[str]) memory_content: the enhanced short-term memories
            (List[float]) memory_importance: the importance scores of the enhanced short-term memories
            (List[List[str]]) insight_content: the insight from the short-term memories
        """
        # if the observation is summarized, otherwise add it into short-term memory
        transfer_flag = False
        existing_memory = [True for _ in range(len(self.short_memories))]
        memory_content, memory_importance, insight_content = [], [], []
        for idx, memory in enumerate(self.short_memories):
            # if exceed the enhancement threshold
            if self.enhance_cnt[idx] >= self.enhance_threshold and existing_memory[idx] is True:
                existing_memory[idx] = False
                transfer_flag = True
                # combine all existing related memories to current memory in short-term memories
                content = [memory]
                # do not repeatedly add observation memory to summary, so use [:-1].
                for enhance_memory in self.enhance_memories[idx][:-1]:
                    content.append(enhance_memory)
                content.append(observation)
                content = ';'.join(content)
                memory_content.append(memory)
                memory_importance.append(self.memory_importance[idx])
                insight = self.get_short_term_insight(content)
                insight_content.append(insight)

        # remove the transferred memories from short-term memories
        if transfer_flag:
            # re-construct the indexes of short-term memories after removing summarized memories
            new_memories = []
            new_embeddings = []
            new_importance = []
            new_enhance_memories = [[] for _ in range(self.capacity)]
            new_enhance_cnt = [0 for _ in range(self.capacity)]
            for idx, memory in enumerate(self.short_memories):
                if existing_memory[idx]:  # True
                    new_enhance_memories[len(new_memories)] = self.enhance_memories[idx]
                    new_enhance_cnt[len(new_memories)] = self.enhance_cnt[idx]
                    new_memories.append(memory)
                    new_embeddings.append(self.short_embeddings[idx])
                    new_importance.append(self.memory_importance[idx])
            self.short_memories = new_memories
            self.short_embeddings = new_embeddings
            self.memory_importance = new_importance
            self.enhance_memories = new_enhance_memories
            self.enhance_cnt = new_enhance_cnt
        return memory_content, memory_importance, insight_content

    def discard_memories(self) -> str:
        """
        discard the least importance memory when short-term memory module exceeds its capacity
        :return: (str) The content of the discard memory
        """
        if len(self.short_memories) > self.capacity:
            memory_dict = dict()
            for idx in range(len(self.short_memories) - 1):
                memory_dict[self.short_memories[idx]] = {'enhance_count': self.enhance_cnt[idx],
                                                         'importance': self.memory_importance[idx]}

            sort_list = sorted(memory_dict.keys(),
                               key=lambda x: (memory_dict[x]['importance'], memory_dict[x]['enhance_count']))
            find_idx = self.short_memories.index(sort_list[0])
            self.enhance_cnt.pop(find_idx)
            self.enhance_cnt.append(0)
            self.enhance_memories.pop(find_idx)
            self.enhance_memories.append([])
            self.memory_importance.pop(find_idx)
            discard_memory = self.short_memories.pop(find_idx)
            self.short_embeddings.pop(find_idx)

            # remove the discard_memory from other short-term memory's enhanced list
            for idx in range(len(self.short_memories)):
                if self.enhance_memories[idx].count(sort_list[0]) != 0:
                    self.enhance_memories[idx].remove(sort_list[0])
                    self.enhance_cnt[idx] -= 1

            return discard_memory

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]):
        """
        Calculate the cosine similarity between two vectors.
        :param embedding1: the first embedding
        :param embedding2: the second embedding
        :return: (float) the cosine similarity
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return similarity


class LongTermMemory():
    """
    Long-term memory is the memory base for the RecAgent.
    """
    llm: LLM
    now: datetime
    memory_retriever: AgentRetriever
    verbose: bool = False
    reflection_threshold: Optional[float] = None
    max_tokens_limit: int = 1000000000
    aggregate_importance: float = 0.0
    decay_rate: float = 0.01
    """The exponential decay factor used as (1.0-decay_rate)**(hrs_passed)."""

    # input keys
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"
    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"

    reflecting: bool = False
    forgetting: bool = False

    forget_num: int = 3

    importance_weight: float = 0.15
    """How much weight to assign the memory importance."""

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    @staticmethod
    def _parse_insight_with_connections(text: str):
        """
        Parse the output of LLM to the insight and the corresponding connections.
        :param text: The output of LLM.
        :return: The insight, and the list of connections.
        """
        pattern = r'\[.*?\]'
        insight = re.sub(pattern, '', text)
        nums = re.findall(r'\d+', text)
        if len(nums) != 0:
            connection_list = list(map(int, nums))
        else:
            connection_list = [0]

        return insight, connection_list

    def _score_memory_importance(self, memory_content: str) -> float:
        """
        Obtain the importance score of this memory.
        :param memory_content: The text of the observation.
        :return: (float) The importance of this observation.
        """
        # prompt = PromptTemplate.from_template(
        #     """
        #     Please give an importance score between 1 to 10 for the following observation. Higher score indicates the observation is more important. More rules that should be followed are
        #     \n(1) The observation that includes entering social media is not important. e.g., David Smith takes action by entering the world of social media.
        #     \n(2) The observation that describes chatting with someone but no specific movie name is not important. e.g., David Smith observed that David Miller expressed interest in chatting about movies.
        #     \n(3) The observation that includes 'chatting' is not important. e.g., David Smith observed that David Miller expressed interest in chatting about movies, indicating a shared passion for films.
        #     \n(4) The observation that includes 'enter the recommender system' is not important. e.g. David Smith enters the Recommender System to explore movie recommendations based on his interests and preferences.
        #     \n(5) The observation that recommends or mentions specific movies is important.
        #     \n(6) More informative indicates more important, especially when two people are chatting.
        #     Please respond with a single integer.
        #     \nObservation:{memory_content}
        #     \nRating:
        #     """
        # )
        prompt = "请根据重要性给以下的记忆打分，分数从0-10，高分意味着更加重要。请回答一个数字。"
        prompt += memory_content
        score = self.llm.invoke(prompt).strip()
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def fetch_memories_with_list(self, observation, stm):
        """
        Transfer the retrieved memories and the enhanced short-term memory with the insight into List.
        :param observation: the observation to retrieve related memories
        :param stm: the short term memory instance
        :return
            (List[(float, str)]) res: the list tuples contains the memory content with corresponding importance score
            (Tuple(List[str], List[float], List[str])) memories_tuple: contains the short-term memories, importances and the insights.
        """
        res_list, memories_tuple = self.fetch_memories(observation, stm=stm)
        res = [(res['importance'], res['content']) for res in res_list]
        return res, memories_tuple

    def fetch_memories(self, observation: str, stm=None):
        """
        Fetch related memories.
        :param observation: the observation to retrieve related memories
        :param stm: the short term memory instance
        :param now: (optional) the current time.
        :return
                (List[Document]) the retrieved memory documents
                (Tuple(List[str], List[float], List[str])) memories_tuple: contains the short-term memories, importances and the insights.
        """
        # reflection do not enhance the short-term memories
        retrieved_list = self.memory_retriever.get_relevant_documents(observation)
        if stm is None:
            return retrieved_list
        # retrieval enhance the short-term memories
        # else:
        #     ltm_memory_list, ltm_importance_scores = [], []
        #     insight_memory_list = []
        #     for document in retrieved_list:
        #         memory_content, memory_importance, insight_content = \
        #             stm.add_stm_memory(document.page_content, document.metadata['importance'], op='Retrieval')
        #         ltm_memory_list.extend(memory_content)
        #         ltm_importance_scores.extend(memory_importance)
        #         insight_memory_list.extend(insight_content)

        #     for idx in range(len(stm.short_memories)):
        #         short_term_document = Document(
        #             page_content=stm.short_memories[idx],
        #             metadata={"importance": stm.memory_importance[idx]}
        #         )
        #         retrieved_list.append(short_term_document)

        #     return retrieved_list, (ltm_memory_list, ltm_importance_scores, insight_memory_list)

    # def format_memories_detail(self, relevant_memories: List[dict]) -> str:
    #     content_strs = set()
    #     content = []
    #     for mem in relevant_memories:
    #         if mem.page_content in content_strs:
    #             continue
    #         content_strs.add(mem.page_content)
    #         created_time = mem.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
    #         content.append(f"- {created_time}: {mem.page_content.strip()}")
    #     return "\n".join([f"{mem}" for mem in content])

    # def format_memories_simple(self, relevant_memories: List[dict]) -> str:
    #     return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    # def _get_memories_until_limit(self, consumed_tokens: int) -> str:
    #     """Reduce the number of tokens in the documents."""
    #     result = []
    #     for doc in self.memory_retriever.memory_stream[::-1]:
    #         if consumed_tokens >= self.max_tokens_limit:
    #             break
    #         consumed_tokens += self.llm.get_num_tokens(doc.page_content)
    #         if consumed_tokens < self.max_tokens_limit:
    #             result.append(doc)
    #     result = self.format_memories_simple(result)
    #     return result

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return similarity

    def _get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        """Return the 1 most salient high-level questions about recent observations."""
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join([o['content'] for o in observations])
        prompt = observation_str
        prompt += "给定上述信息，我们可以提出的最突出的高级问题是什么？请回答一个问题。"
        result = self.llm.invoke(prompt).strip()
        return self._parse_list(result)

    def _get_insights_on_topic(
            self, topic: str, now: Optional[datetime] = None
    ):
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join(
            [
                # f"{i + 1}. {memory.page_content}"
                memory['content']
                for i, memory in enumerate(related_memories)
            ]
        )
        prompt = "给定一些相关的记忆，请回答一个问题。\n" \
                 "记忆：\n"
        prompt += related_statements
        prompt += "\n问题:\n"
        prompt += topic
        prompt += "请确定一个主要insight来回答上述问题，同时指定insight来自哪些记忆。请严格遵循以下格式：insight的内容\n相关记忆。" \
                "（insight可以从多个记忆中派生得出。insight需要在句子结构和内容上与陈述有很大不同。）"

        result = self.llm.invoke(prompt).strip()

        result_insight = self._parse_list(result_insight)
        result_insight = [self._parse_insight_with_connections(res) for res in result_insight]
        statements_id = result_insight[0][1]

        pattern = r"(?<=\[)\d+(?=\])"
        indexes = []
        embedding_1 = self.llm.get_embedding(result_insight[0][0])
        for memory_id in statements_id:
            if memory_id < 0 or memory_id >= len(self.memory_retriever.memory_stream):
                continue
            memory = self.memory_retriever.memory_stream[memory_id]['content']
            if memory == '[MERGE]' or memory == '[FORGET]':
                continue
            memory_embedding = self.llm.get_embedding(memory)
            similarity = self.cosine_similarity(embedding_1, memory_embedding)
            # Sigmoid function
            value = 1 / (1 + np.exp(-similarity))
            if value >= 0.72:
                match = re.search(pattern, memory)
                idx = match.group()
                indexes.append(int(idx))

        for idx in indexes:
            self.memory_retriever.memory_stream[idx]['content'] = '[MERGE]'
            self.memory_retriever.memory_stream[idx]['importance'] = 1.0
            self.memory_retriever.memory_stream[idx]['last_accessed_at'] = self.now

        return result_insight

    def pause_to_reflect(self, now: Optional[datetime] = None):
        """
        Reflect on recent observations and generate 'insights'.
        :param now: (optional) The current time.
        :return: The list of new insights. [No use for this version.]
        """
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(topic, now=now)
            for insight in insights:
                text, par_list = insight
                importance_cur, recency_cur = 0.0, 0.0
                valid = 0
                for par in par_list:
                    if par < len(self.memory_retriever.memory_stream):
                        importance_cur += self.memory_retriever.memory_stream[par]['importance']
                        valid += 1
                if valid == 0:
                    importance_cur = 0.0
                else:
                    importance_cur /= valid
                ltm = importance_cur, now, text
                self.add_memory(ltm, now=now)
            new_insights.extend(insights)

        return new_insights

    def obtain_forget_prob_list(self):
        """
        Obtain the forgetting probability of each memory.
        :return: (List) The distribution of forgetting probability.
        """

        def score_func(importance, last_accessed_time):
            """
            Given the importance score and last accessed time, calculate the score of this memory.
            :param importance: The importance score.
            :param last_accessed_time: The last accessed time.
            :return: Score of this memory.
            """
            hours_passed = (self.now - last_accessed_time).total_seconds() / 3600
            recency = (1.0 - self.decay_rate) ** hours_passed

            return max(recency ** 1.5, 0.01) * (importance + recency) / 2

        score_list = []
        for ind, mem in enumerate(self.memory_retriever.memory_stream):
            score = score_func(mem['importance'], mem['last_accessed_at'])
            score_list.append(score)
        score_list = 1.0 - np.array(score_list)
        return score_list / np.sum(score_list)

    def pause_to_forget(self):
        """
        Forget parts of long term memories.
        """
        if self.verbose:
            logger.info("Character is forgetting.")

        prob_list = self.obtain_forget_prob_list()
        if len(prob_list) != 0:
            for idx in range(len(prob_list)):
                if (self.now - self.memory_retriever.memory_stream[idx]['last_accessed_at']).total_seconds() / 3600 <= 24:
                    continue
                if random() < prob_list[idx]:
                    self.memory_retriever.memory_stream[idx]['content'] = '[FORGET]'
                    self.memory_retriever.memory_stream[idx]['importance'] = 1.0
                    self.memory_retriever.memory_stream[idx]['last_accessed_at'] = self.now

    def add_memory(self, ltm, now=None):
        """
        Store the long term memory.
        :param ltm: The long term memory that is ready to be stored.
        :param now: Current time.
        :return: List of IDs of the added texts. [No use in this version.]
        """
        importance, last_accessed_at, text = ltm
        if not self.reflecting:
            self.aggregate_importance += importance
        memory_idx = len(self.memory_retriever.memory_stream)
        document = {
            "content": "[%d]".format(memory_idx) + text,
            "importance": importance,
            "last_accessed_at": last_accessed_at,
        }
        result = self.memory_retriever.add_documents([document], current_time=now)
        return result

    def save_context(self, inputs: Dict[str, Any], ltm_list: list) -> None:
        """
        Store the long term memories. Execute reflection and forgetting.
        :param inputs: [No use for this version.]
        :param ltm_list: The list of long term memory with tuple format (importance score[float], now[datetime], memory[string]).
        :return: None
        """
        now = self.now
        for ltm in ltm_list:
            self.add_memory(ltm, now)
        # When the aggregation of importance is above the threshold, execute the reflection function once.
        if (
                self.reflection_threshold is not None
                and self.aggregate_importance > self.reflection_threshold
                and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        # Execute the forget function once.
        if True:
            self.forgetting = True
            self.pause_to_forget()
            self.forgetting = False

    # def print_memory(self):
    #     """
    #     [Tool for Debug] Print the long term memories.
    #     """
    #     for ind, mem in enumerate(self.memory_retriever.memory_stream):
    #         hours_passed = (self.now - mem['last_accessed_at']).total_seconds() / 3600
    #         recency = (1.0 - self.decay_rate) ** hours_passed
    #         print('[%d] (importance: %f, recency: %f) %s' % (
    #             ind, mem['importance'], recency, mem['content']))

    def update_now(self, now: datetime):
        """
        Update the current time.
        :param now: Current time.
        """
        self.now = now
        self.memory_retriever.now = now

    def clear(self) -> None:
        """
        Clear all the memories in long term memory.
        """
        self.memory_retriever.memory_stream = []


class WorkingMemory():
    pass


class AgentMemory():
    """
    RecAgentMemory is the proposed memory module for RecAgent. We replace `GenerativeAgentMemory` with this class.
    Similarly, it has three necessary methods to implement:
    - load_memory_variables: given inputs, return the corresponding information in the memory.
    - save_context: accept observations and store them as memory.
    - clear: clear the memory content.

    We have three key components, which is consistent with human's brain.
    - SensoryMemory: Receive observations, abstract significant information, and pass to short-term memory.
    - ShortTermMemory: Receive sensory memories, enhance them with new observations or retrieved memories,
                       and then transfer the enhanced short-term memories with an insight to long-term memory,
                       or discard the less important memory in cases of capacity overload.
    - LongTermMemory: Receive short-term memories, store and forget memories, and retrival memories to short-term memory.

    """
    llm: LLM = None
    verbose: bool = False
    now: datetime = None

    sensoryMemory: SensoryMemory = None
    shortTermMemory: ShortTermMemory = None
    longTermMemory: LongTermMemory = None

    importance_weight: float = 0.9
    """How much weight to assign the memory importance."""

    # input keys
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"
    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"

    def __init__(self, llm, memory_retriever, now, verbose=False, reflection_threshold=None):
        super(AgentMemory, self).__init__()

        self.llm = llm
        self.now = now
        self.sensoryMemory = SensoryMemory(llm)
        self.shortTermMemory = ShortTermMemory(llm)
        self.longTermMemory = LongTermMemory(llm=llm, memory_retriever=memory_retriever, now=self.now, verbose=verbose,
                                             reflection_threshold=reflection_threshold)

    @property
    def memory_variables(self) -> List[str]:
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return 'most_recent_memories' with fetched memories (not recent memories).
        :param inputs: The dict that contains the key 'observation'.
        :return: The fetched memories.
        """

        ltm_memory_list, memories_tuple = self.longTermMemory.fetch_memories_with_list(inputs['observation'],
                                                                                       self.shortTermMemory)
        self.save_context_after_retrieval(memories_tuple)
        if len(ltm_memory_list) == 0:
            memory_tmp = ''
        else:
            memory_tmp = [memory[1] for memory in ltm_memory_list]
        memory_tmp = ''.join(memory_tmp)
        output = {'most_recent_memories': memory_tmp}
        return output

    def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        # prompt = PromptTemplate.from_template(
        #     """
        #     Please give an importance score between 1 to 10 for the following observation. Higher score indicates the observation is more important. More rules that should be followed are
        #     \n(1) The observation that includes entering social media is not important. e.g., David Smith takes action by entering the world of social media.
        #     \n(2) The observation that describes chatting with someone but no specific movie name is not important. e.g., David Smith observed that David Miller expressed interest in chatting about movies.
        #     \n(3) The observation that includes 'chatting' is not important. e.g., David Smith observed that David Miller expressed interest in chatting about movies, indicating a shared passion for films.
        #     \n(4) The observation that includes 'enter the recommender system' is not important. e.g. David Smith enters the Recommender System to explore movie recommendations based on his interests and preferences.
        #     \n(5) The observation that recommends or mentions specific movies is important.
        #     \n(6) More informative indicates more important, especially when two people are chatting.
        #     Please respond with a single integer.
        #     \nObservation:{memory_content}
        #     \nRating:
        #     """
        # )
        prompt = "请给下述记忆打分，分数从0-10，高分意味着更加重要。请回答一个数字。"
        prompt += memory_content
        score = self.llm.invoke(prompt).strip()
        if self.verbose:
            logger.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def add_memory(self, memory_content: str, now: Optional[datetime] = None):
        """
        The Simulator can add memory by using this function.
        :param memory_content: The content of memory.
        :param now: Current time.
        """
        self.save_context(
            {},
            {
                self.add_memory_key: memory_content,
                self.now_key: now,
            },
        )

    def save_context_after_retrieval(self, memories_tuple):
        """
        The RecAgent can transfer short-term memory to long-term memory.
        :param memories_tuple:  (Tuple(List[str], List[float], List[str])) memories_tuple: contains the short-term memories, importances and the insights.
        :return None
        """
        ltm_memory_list, ltm_importance_scores, insight_memory_list = memories_tuple
        insight_memory_list = [memory[0] for memory in insight_memory_list]
        insight_scores_list = [self._score_memory_importance(memory) for memory in insight_memory_list]

        all_memories = ltm_memory_list + insight_memory_list
        all_memory_scores = ltm_importance_scores + insight_scores_list
        save_ltm_memory = [(all_memory_scores[i], self.now, all_memories[i])
                           for i in range(len(all_memories))]
        self.longTermMemory.save_context({}, save_ltm_memory)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        The RecAgent can add memory by using this function.
        :param inputs: Will be directly pass to LongTermMemory. [No use for this version.]
        :param outputs: The core memory dict that is passed from RecAgent. It has to obtain the key 'add_memory' to save the memory content.
        :return: None
        """
        # If the 'outputs' does not contain the memory, then exit the function.
        if self.add_memory_key not in outputs:
            return
        # Add the observation into the buffer of sensory memory, and obtain a list of short term memory if the buffer is full.
        obs = outputs[self.add_memory_key]
        stm_memory_list = self.sensoryMemory.add_sensory_memory(obs)
        if stm_memory_list is None:
            return
        else:
            ltm_memory_list, ltm_importance_scores = [], []
            insight_memory_list = []
            for stm_memory in stm_memory_list:
                memory_content, memory_importance, insight_content \
                    = self.shortTermMemory.add_short_memory(stm_memory[1], stm_memory[0], op='add')
                ltm_memory_list.extend(memory_content)
                ltm_importance_scores.extend(memory_importance)
                insight_memory_list.extend(insight_content)

            insight_memory_list = [memory[0] for memory in insight_memory_list]
            insight_scores_list = [self._score_memory_importance(memory) for memory in insight_memory_list]

            all_memories = ltm_memory_list + insight_memory_list
            all_memory_scores = ltm_importance_scores + insight_scores_list
            save_ltm_memory = [(all_memory_scores[i], self.now, all_memories[i])
                               for i in range(len(all_memories))]
            # Store the long term memories.
            self.longTermMemory.save_context(inputs, save_ltm_memory)

    def update_now(self, now: datetime):
        """
        Update the current time.
        :param now: Current time.
        """
        self.now = now
        self.longTermMemory.update_now(self.now)

    def clear(self) -> None:
        """
        Clear all the (long term) memory in RecAgentMemory.
        """
        self.longTermMemory.clear()

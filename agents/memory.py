import re
import torch
import random
import logging
import numpy as np
from typing import List
from datetime import datetime

from llm.llm import LLM
from llm.prompt import get_prompt

logger = logging.getLogger("MyLogger")


def score_memory_importance(memory: str, llm: LLM) -> float:
    prompt_path = "llm/prompt_template/score_memory_importance.txt"
    prompt = get_prompt(prompt_path, {
        "memory": memory
    })
    result = llm.invoke(prompt).strip()
    logger.debug(prompt + "\n\n" + result)
    match = re.search(r"^\D*(\d+)", result)
    if match:
        return float(match.group(1)) / 10
    else:
        logger.error("Score memory importance fialure: %s" % (result))
        return 0.0

def get_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


class MemoryRetriever():
    def __init__(self, llm: LLM):
        self.llm = llm
        self.recency_decay = 0.99
        self.recency_weight = 0.3
        self.relevance_weight = 0.4
        self.importance_weight = 0.3
        self.top_n = 5

    def get_combined_score(self, relevance: float, importance: float, last_access_time: datetime, now: datetime) -> float:
        return relevance * self.relevance_weight + \
            (self.recency_decay ** (now - last_access_time).total_seconds() / 3600) * self.recency_weight + \
            importance * self.importance_weight

    def get_relevant_memories(self, query: str, memories: List[dict], now: datetime, top_n: int = 0) -> List[dict]:
        if len(memories) == 0:
            return []
        top_n = top_n if not top_n == 0 else self.top_n # if top_n isn't given, set it to self.top_n
        if query: # sort with query
            query_embedding = self.llm.embedding_invoke(query)
            docs = [
                (memory, \
                 self.get_combined_score(get_cosine_similarity(query_embedding, memory["embedding"]), memory["importance"], memory["last_access_time"], now))
                for memory in memories if memory["text"] != "[FORGET]"
            ]
        else: # sort without query
            docs = [
                (memory, \
                 self.get_combined_score(0, memory["importance"], memory["last_access_time"], now))
                for memory in memories if memory["text"] != "[FORGET]"
            ]
        docs.sort(key=lambda x: x[1], reverse=True)

        result_memories = []
        for memory, _ in docs:
            if len(result_memories) < top_n:
                memory["last_access_time"] = now
                result_memories.append({
                    "text": memory["text"],
                    "importance": memory["importance"],
                    "embedding": memory["embedding"]
                })
        return result_memories

    def get_forget_socre(self, importance: float, last_access_time: datetime, now: datetime) -> float:
        recency = self.recency_decay ** ((now - last_access_time).total_seconds() / 3600)
        return max(recency ** 1.5, 0.01) * (importance + recency) / 2


class SensoryMemory():
    def __init__(self, llm, capacity=1):
        self.llm: LLM = llm
        self.capacity: int = capacity
        self.buffer: List = []

    def clear(self):
        self.buffer.clear()

    def add_sensory_memory(self, observation: str):
        self.buffer.append(observation)
        if len(self.buffer) >= self.capacity:
            return self.transfer_to_short()
        else:
            return []

    def transfer_to_short(self):
        prompt_path = "llm/prompt_template/sensory_to_short.txt"
        prompt_input = {
            "sensory_memories": '\n'.join(["[%d] %s" % (ind, obs) for ind, obs in enumerate(self.buffer)])
        }
        prompt = get_prompt(prompt_path, prompt_input)
        result = self.llm.invoke(prompt).strip()
        # TODO parse the result
        logger.debug(prompt + "\n\n" + result)
        stm_memories = [{
            "text": result,
            "importance": score_memory_importance(result, self.llm),
        }]
        self.clear()
        return stm_memories
    
    def save_to_dict(self):
        return {
            "buffer": self.buffer
        }


class ShortTermMemory():
    def __init__(self, llm: LLM):
        self.llm: LLM = llm
        self.capacity: int = 10
        self.short_memories: List[dict] = []
        self.enhance_cnts: List[int] = []
        self.enhance_memory_texts: List[List[str]] = []
        self.enhance_threshold: int = 3
        self.engance_prob_threshold: float = 0.7

    def add_short_memory(self, new_memory_text: str, importance: float, op: str):
        new_memory_embedding = self.llm.embedding_invoke(new_memory_text)
        ltm_memories, ltm_insights = self.transfer_to_long(new_memory_text, new_memory_embedding)
        if op == 'Add': # else Retrival
            self.short_memories.append({
                "text": new_memory_text,
                "importance": importance,
                "embedding": new_memory_embedding
            })
            self.enhance_cnts.append(0)
            self.enhance_memory_texts.append([])
            self.discard_memories()
        return ltm_memories + ltm_insights

    def transfer_to_long(self, memory_text: str, memory_embedding):
        ltm_memories, ltm_insights = [], []
        transfer_records = [False for _ in range(len(self.short_memories))]

        primacy_const = 0.1
        for idx, memory in enumerate(self.short_memories):
            similarity = get_cosine_similarity(memory_embedding, memory["embedding"])
            if idx + 1 == len(self.short_memories): # primacy effect
                similarity = min(similarity + primacy_const, 1.0)
            enhance_prob = 1 / (1 + np.exp(-similarity))
            if not (enhance_prob >= self.engance_prob_threshold and random.random() <= enhance_prob): # else enhance
                continue
            self.enhance_cnts[idx] += 1
            self.enhance_memory_texts[idx].append(memory_text)
            # transfer to long term memory
            if self.enhance_cnts[idx] >= self.enhance_threshold:
                enhance_texts = [memory["text"]] + self.enhance_memory_texts[idx]
                insights = self.get_short_term_insight(enhance_texts)
                ltm_insights.extend([
                    {
                        "text": insight,
                        "importance": score_memory_importance(insight, self.llm),
                        "embedding": self.llm.embedding_invoke(insight),
                    }
                    for insight in insights
                ])
                ltm_memories.append({
                    "text": memory["text"],
                    "importance": memory["importance"],
                    "embedding": memory["embedding"],
                })
                transfer_records[idx] = True
        
        if len(ltm_memories) != 0:
            self.short_memories = [memory for idx, memory in enumerate(self.short_memories) if not transfer_records[idx]]
            self.enhance_cnts = [cnt for idx, cnt in enumerate(self.enhance_cnts) if not transfer_records[idx]]
            self.enhance_memory_texts = [texts for idx, texts in enumerate(self.enhance_memory_texts) if not transfer_records[idx]]

        return ltm_memories, ltm_insights

    def get_short_term_insight(self, memories_text: List[str]):
        prompt_path = "llm/prompt_template/short_term_insight.txt"
        prompt_input = {
            "stm_memories": '\n'.join("[%d] %s" % (ind, mem) for ind, mem in enumerate(memories_text))
        }
        prompt = get_prompt(prompt_path, prompt_input)
        result = self.llm.invoke(prompt).strip()
        logger.debug(prompt + "\n\n" + result)
        # TODO parse the result
        return [result]

    def discard_memories(self) -> str:
        def discard_score(importance: float, enhance_cnt: int) -> float: # the higher more likely to be discarded
            return -(importance * np.sqrt(enhance_cnt + 1))
        if len(self.short_memories) <= self.capacity:
            return None
        docs = [
            (idx, discard_score(self.short_memories[idx]["importance"], self.enhance_cnts[idx]))
            for idx in range(len(self.short_memories))
        ]
        docs.sort(key=lambda x: x[1], reverse=True)
        discard_idx = docs[0][0]
        discard_memory = self.short_memories.pop(discard_memory)
        self.enhance_cnts.pop(discard_idx)
        self.enhance_memory_texts.pop(discard_idx)
        for idx in range(len(self.short_memories)):
            if self.enhance_memory_texts[idx].count(discard_memory["text"]) != 0:
                self.enhance_memory_texts[idx].remove(discard_memory["text"])
                self.enhance_cnts[idx] -= 1

        return discard_memory

    def clear(self):
        self.short_memories.clear()
        self.enhance_cnts.clear()
        self.enhance_memory_texts.clear()

    def save_to_dict(self, with_embedding: bool = True):
        if with_embedding:
            return {
                "short_memories": self.short_memories,
                "enhance_cnts": self.enhance_cnts,
                "enhance_memory_texts": self.enhance_memory_texts
            }
        else:
            return {
                "short_memories": [
                    {
                        "text": memory["text"],
                        "importance": memory["importance"]
                    }
                    for memory in self.short_memories
                ],
                "enhance_cnts": self.enhance_cnts,
                "enhance_memory_texts": self.enhance_memory_texts
            }


class LongTermMemory():
    def __init__(self, llm: LLM, memory_retriever: MemoryRetriever, reflection_threshold: float):
        self.llm: LLM = llm
        self.long_memories: List[dict] = []
        self.memory_retriever: MemoryRetriever = memory_retriever
        self.reflection_threshold: float = reflection_threshold
        self.aggregate_importance: float = 0.0

    def add_long_memory(self, ltm_memories: List[dict], now: datetime):
        for ltm_memory in ltm_memories:
            self.long_memories.append({
                "text": ltm_memory["text"],
                "importance": ltm_memory["importance"],
                "embedding": ltm_memory["embedding"],
                "last_access_time": now
            })
            self.aggregate_importance += ltm_memory["importance"]
        if self.aggregate_importance >= self.reflection_threshold:
            reflect_insights = self.reflect(now)
            for reflect_insight in reflect_insights:
                self.long_memories.append({
                    "text": reflect_insight,
                    "importance": score_memory_importance(reflect_insight, self.llm),
                    "embedding": self.llm.embedding_invoke(reflect_insight),
                    "last_access_time": now
                })
            self.aggregate_importance = 0.0
            self.forget(now)

    def get_topics_for_reflect(self, last_k: int = 10) -> List[str]:
        prompt_path = "get_topic_for_reflect.txt"
        memories = self.long_memories[-last_k:]
        prompt_input = {
            "memories": "\n".join(["%d. " % idx + memory["text"] for idx, memory in enumerate(memories)])
        }
        prompt = get_prompt(prompt_path, prompt_input)
        result = self.llm.invoke(prompt).strip()
        logger.debug(prompt + "\n\n" + result)
        # TODO parse the result
        return [result]

    def get_insights_on_topic(self, topic: str, now:datetime):
        related_memories, _ = self.fetch_memories(topic, now)
        prompt_path = "llm/prompt_template/get_insight_on_topic.txt"
        prompt_input = {
            "related_memories": "\n".join(["%d. " % idx + memory["text"] for idx, memory in enumerate(related_memories)])
        }
        prompt = get_prompt(prompt_path, prompt_input)

        result = self.llm.invoke(prompt).strip()
        logger.debug(prompt + "\n\n" + result)
        
        return [result]

    def reflect(self, now: datetime):
        reflect_insights = []
        topics = self.get_topics_for_reflect()
        for topic in topics:
            insights = self.get_insights_on_topic(topic, now)
            reflect_insights.extend(insights)
        return reflect_insights

    def get_forget_probs(self, now: datetime):
        probs = [self.memory_retriever.get_forget_socre(memory["importance"], memory["last_access_time"], now) \
                 for memory in self.long_memories]
        probs = 1.0 - np.array(probs)
        return probs / np.sum(probs)

    def forget(self, now: datetime):
        probs = self.get_forget_probs(now)
        for idx in range(len(probs)):
            if (now - self.long_memories[idx]["last_access_time"]).total_seconds() / 3600 <= 24: # within one day
                continue
            if random.random() < probs[idx]:
                self.long_memories[idx]["text"] = "[FORGET]"
                self.long_memories[idx]["importance"] = 1.0
                self.long_memories[idx]["embedding"] = None
                self.long_memories[idx]["last_access_time"] = now

    def fetch_memories(self, query: str, now:datetime, stm: ShortTermMemory = None):
        # reflection do not enhance the short-term memories
        retrieved_memories = self.memory_retriever.get_relevant_memories(query, self.long_memories, now)
        if stm is None:
            return retrieved_memories, []
        # retrieval enhance the short-term memories
        else:
            stm_memories_copy = [
                {
                    "text": memory["text"],
                    "importance": memory["importance"],
                    "embedding": memory["embedding"]
                }
                for memory in stm.short_memories
            ]
            ltm_memories = []
            for retrieved_memory in retrieved_memories:
                ltm_memories.extend(stm.add_short_memory(retrieved_memory["text"], retrieved_memory["importance"], op="Retrieval"))
            # contain short term memories
            retrieved_memories.extend(stm_memories_copy)
            return retrieved_memories, ltm_memories

    def clear(self) -> None:
        self.long_memories.clear()

    def save_to_dict(self, with_embedding: bool = True):
        if with_embedding:
            return {
                "long_memories": [
                    {
                        "text": memory["text"],
                        "importance": memory["importance"],
                        "embedding": memory["embedding"],
                        "last_access_time": memory["last_access_time"].strftime("%Y-%m-%d %H:%M:%S")
                    }
                    for memory in self.long_memories
                ],
                "aggregate_importance": self.aggregate_importance
            }
        else:
            return {
                "long_memories": [
                    {
                        "text": memory["text"],
                        "importance": memory["importance"],
                        "last_access_time": memory["last_access_time"].strftime("%Y-%m-%d %H:%M:%S")
                    }
                    for memory in self.long_memories
                ],
                "aggregate_importance": self.aggregate_importance
            }


class WorkingMemory():
    pass


class AgentMemory():
    def __init__(self, llm, memory_retriever: MemoryRetriever, reflection_threshold:float = 50):
        self.llm: LLM = llm
        self.sensoryMemory = SensoryMemory(llm)
        self.shortTermMemory = ShortTermMemory(llm)
        self.longTermMemory = LongTermMemory(llm=llm, memory_retriever=memory_retriever,
                                                reflection_threshold=reflection_threshold)
    
    def add_memory(self, memory_text: str, now: datetime):
        stm_memories = self.sensoryMemory.add_sensory_memory(memory_text)
        if stm_memories: # If stm_memories is not empty, transfer to short term memory
            ltm_memories = []
            for stm_memory in stm_memories:
                ltm_memories.extend(self.shortTermMemory.add_short_memory(stm_memory["text"], stm_memory["importance"], op='Add'))
            # Transfer short term memory to the long term memories.
            self.longTermMemory.add_long_memory(ltm_memories, now)

    def retrive_memories(self, query: str, now: datetime) -> List[dict]:
        retrieved_memories, ltm_memories = self.longTermMemory.fetch_memories(query, now, self.shortTermMemory)
        self.longTermMemory.add_long_memory(ltm_memories, now)
        return retrieved_memories

    def clear(self) -> None:
        self.sensoryMemory.clear()
        self.shortTermMemory.clear()
        self.longTermMemory.clear()

    def save_to_dict(self, with_embedding: bool = True):
        return {
            "sensoryMemory": self.sensoryMemory.save_to_dict(),
            "shortTermMemory": self.shortTermMemory.save_to_dict(with_embedding=with_embedding),
            "longTermMemory": self.longTermMemory.save_to_dict(with_embedding=with_embedding)
        }
    
    def load_from_dict(self, memory_dict: dict) -> None:
        self.sensoryMemory.buffer = memory_dict["sensoryMemory"]["buffer"]
        self.shortTermMemory.short_memories = memory_dict["shortTermMemory"]["short_memories"]
        self.shortTermMemory.enhance_cnts = memory_dict["shortTermMemory"]["enhance_cnts"]
        self.shortTermMemory.enhance_memory_texts = memory_dict["shortTermMemory"]["enhance_memory_texts"]
        self.longTermMemory.long_memories = [
            {
                "text": memory["text"],
                "importance": memory["importance"],
                "embedding": memory["embedding"],
                "last_access_time": datetime.strptime(memory["last_access_time"], "%Y-%m-%d %H:%M:%S")
            }
            for memory in memory_dict["longTermMemory"]["long_memories"]
        ]
        self.longTermMemory.aggregate_importance = memory_dict["longTermMemory"]["aggregate_importance"]


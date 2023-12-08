import json
from datetime import datetime
from functools import cache
import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llm.llm import LLM
from llm.prompt import get_prompt

class SocialNetAgent:

    def __init__(self, user_dic: dict, llm: LLM):
        # self.name = user_dic["name"]                # nickname of the agent
        # self.description = user_dic["description"]  # description of the agent
        # self.gender = user_dic["gender"]            # gender of the agent
        # self.location = user_dic["location"]        # location of the agent
        self.user_prof = user_dic                     # user dictionary
        self.memory = []
        self.llm = llm

    def add_to_memory(self, observation, curr_time, calculate_importance=False, **kwargs):
        memory_item = {
            "creation_time": curr_time, 
            "observation": observation, 
            "last_access_time": curr_time,
            "importance": 5 if not calculate_importance else SocialNetAgent.calculate_importance(observation),
        }
        for arg in kwargs:
            memory_item[arg] = kwargs[arg]

        self.memory.append(memory_item)

    def get_relevance_scores(self, query):
        query_embedding = self.llm.embedding_invoke(query)
        logging.info(json.dumps([memory_item["activity"] for memory_item in self.memory], indent=4))
        memory_item_embeddings = [self.llm.embedding_invoke(memory_item["observation"]) for memory_item in self.memory]
        scores = cosine_similarity([query_embedding], memory_item_embeddings)[0]
        return scores

    def calculate_recency_score(self, time0, time1):
        duration_hours = (time1 - time0).total_seconds() // 3600
        score = 0.99**duration_hours
        return score

    def min_max_scaling(self, scores):
        # if min == max, all scores == 1
        min_score = min(scores)
        max_score = max(scores)
        scaled_scores = [(score-min_score+1e-10) / (max_score-min_score+1e-10) for score in scores]
        return scaled_scores

    def combine_scores(self, relevance_scores, importance_scores, recency_scores, relevance_alpha=1, importance_alpha=1, recency_alpha=1):
        combined_scores = []
        for i in range(len(relevance_scores)):
            combined_score = relevance_scores[i] * relevance_alpha
            combined_score += importance_scores[i] * importance_alpha
            combined_score += recency_scores[i] * recency_alpha
            combined_scores.append(combined_score)
        return combined_scores

    def retrieve_memories(self, query, curr_time, top_n=5, timestamp=False):
        relevance_scores = self.get_relevance_scores(query)
        importance_scores = [memory_item["importance"] for memory_item in self.memory]
        recency_scores = [self.calculate_recency_score(memory_item["last_access_time"], curr_time) for memory_item in self.memory]
        combined_scores = self.combine_scores(
            self.min_max_scaling(relevance_scores), 
            self.min_max_scaling(importance_scores), 
            self.min_max_scaling(recency_scores)
        )
        ordered_data = np.argsort(combined_scores)[::-1]
        retrieval_memory_indices = ordered_data[:top_n]
        memory_statements = [self.memory[i]['obserbation'] for i in retrieval_memory_indices]
        return memory_statements

    def get_questions_for_reflection(self):
        # prompt = ", ".join([memory_item["observation"] for memory_item in self.memory[-100:]])
        # prompt += "Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?"
        prompt = "to do"
        questions = self.llm.invoke(prompt, max_tokens=100)
        question_list = [question + "?" for question in questions.split("?")]
        return question_list
    
    def reflect(self, curr_time):
        questions = self.get_questions_for_reflection()
        for question in questions:
            memories = self.retrieve_memories(question, curr_time, top_n=15)
            prompt = f"Statements about {self.name}\n"
            id2memory = {str(i): memories[i] for i in range(len(memories))}
            for i, memory in enumerate(memories):
                prompt += f"{i}. {memory}\n"
            prompt += "What 5 high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))"
            insights = self.llm.invoke(prompt)
            # remove the 1. 2. or 3. 
            insights_list = [' '.join(insight.strip().split(' ')[1:]) for insight in insights.split(")")][:5]
            for insight in insights_list:
                insight_pair = insight.split("(")
                insight_only, reason = insight_pair
                source_nodes = [node.strip() for node in reason.replace(' ', ",").split(",") if node.strip().isnumeric()]
                source_memories = [id2memory[source_node] for source_node in source_nodes]
                self.add_to_memory(
                    activity=insight_only.strip(), 
                    curr_time=curr_time,
                    source_memories=source_memories,
                    memory_type="reflect"
                )
        return insights

    @cache
    def calculate_importance(self, memory_statement):
        #example memory statement -  buying groceries at The Willows Market and Pharmacy
        prompt = f'''
        On the scale of 1 to 10, where 1 is purely mundane
        (e.g., brushing teeth, making bed) and 10 is
        extremely poignant (e.g., a break up, college
        acceptance), rate the likely poignancy of the
        following piece of memory.
        Memory: {memory_statement}
        Rating:'''
        return int(self.llm.invoke(prompt, max_tokens=1))

    def get_agent_information(self, aspect="core characteristics", curr_time=None):
        memory_query = f"{self.name}'s {aspect}"
        memory_statements = self.retrieve_memories(memory_query, curr_time)
        joined_memory_statements = '\n- '.join(memory_statements)
        prompt = f"""How would one describe {memory_query} given the following statements?\n- {joined_memory_statements}"""
        return self.llm.invoke(prompt)

    @cache
    def get_agent_summary_description(self, curr_time):
        """
        In our implementation, this summary comprises agents’
        identity information (e.g., name, age, personality), as well as a
        description of their main motivational drivers and statements that
        describes their current occupation and self-assessment.

        This is currently cached using the key curr_time, but can be cached based on the day or hour
        """
        core_characteristics = self.get_agent_information(aspect="core characteristics", curr_time=curr_time)
        current_daily_occupation = self.get_agent_information(aspect="current daily occupation", curr_time=curr_time)
        feelings = self.get_agent_information(aspect="feeling about his recent progress in life", curr_time=curr_time)

        description = f"""
        Name: {self.name} (age: {self.age})
        Innate traits: {', '.join(self.traits)}
        {core_characteristics}
        {current_daily_occupation}
        {feelings}
        """
        return description
    
    def reaction_to_post(self, post, curr_time):
        # When recieving a post, the agent will decide whether to repost or follow the poster
        prompt = f"""
        {self.name} is a person whose description is as follows:
        {self.get_agent_summary_description(curr_time)}
        {self.name} received the following post:
        {post}
        {self.name} can choose to do both, one, or neither of the following:
        1. Repost the post
        2. Follow the poster
        """
        response = self.llm.invoke(prompt)
        return response
    
    def generate_post(self):
        # When generating a post, the agent will decide what to post
        prompt_template_path = "llm/prompt_template/post_generation.txt"
        prompt_input = self.user_prof.copy()
        prompt_input["token_limit"] = 250
        prompt = get_prompt(prompt_template_path, prompt_input)
        logging.info("Prompt: " + prompt)
        response = self.llm.invoke(prompt)
        post = self.llm.parse_response(response, "post")
        return post
    
    def react_to_post(self, msg):
        # When recieving a message, the agent will decide how to react
        prompt_template_path = "llm/prompt_template/react_to_post.txt"
        prompt_input = self.user_prof.copy()
        prompt_input["post"] = msg["content"]
        prompt = get_prompt(prompt_template_path, prompt_input)
        logging.info("Prompt: " + prompt)
        response = self.llm.invoke(prompt)
        reaction = self.llm.parse_response(response, "react")
        return reaction

    @staticmethod
    def parse_reaction_response(response):
        """
        The first sentence should either contain Yes or No (and maybe some additional words). If yes, the second sentence onwards tells of the actual reaction
        """
        response_parts = response.split(".")
        if "yes" in response_parts[0].lower() and len(response_parts) > 1:
            full_response = '.'.join(response_parts[1:])
            return full_response
        return None

    @staticmethod
    def convert_conversation_in_linearized_representation(conversation_history):
        linearized_conversation_history = 'Here is the dialogue history:\n'
        for turn in conversation_history:
            #sometime conversation last turn has no text
            if turn['text'] is not None:
                linearized_conversation_history += f"{turn['name']}: {turn['text']}\n"
        return linearized_conversation_history


import torch
import numpy as np
import logging
import zhipuai
import openai
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline

zhipuai.api_key = "78a2dfa223061c83018dd3e89b4b09ed.hF7ap5HXcXE8qtMp"

logger = logging.getLogger("MyLogger")


class LLM():
    def __init__(self):
        raise NotImplementedError

    def invoke(self, prompt):
        raise NotImplementedError

    def async_invoke(self, prompt):
        raise NotImplementedError
    
    def embedding_invoke(self, text) -> np.ndarray:
        raise NotImplementedError
    
    def embedding_async_invoke(self, text):
        raise NotImplementedError
    
    @staticmethod
    def parse_response(response):
        raise NotImplementedError


class ZhipuAi(LLM):
    def __init__(self, api_key):
        self.turbo_model = "chatglm_turbo"
        self.embedding_model = "text_embedding"

    # def invoke(self, prompt, top_p=0.7, temperature=0.95):
    #     response = zhipuai.model_api.invoke(
    #         model=self.turbo_model,
    #         prompt=[{"role": "user", "content": prompt}],
    #         top_p=0.7, # default 0.7
    #         temperature=0.9, # default 0.95
    #     )
    #     print(response)
    #     return response

    def invoke(self, prompt, top_p=0.7, temperature=0.95):
        response = zhipuai.model_api.sse_invoke(
            model=self.turbo_model,
            prompt=[{"role": "user", "content": prompt}],
            top_p=0.7, # default 0.7
            temperature=0.9, # default 0.95
        )
        print(response)
        result = ""
        for event in response.events():
            if event.event == "add":
                result += event.data
            elif event.event == "error" or event.event == "interrupted":
                result += event.data
            if event.event == "finish":
                result += event.data
                print(event.meta)
        return result

    def async_invoke(self, prompt, top_p=0.7, temperature=0.95):
        response = zhipuai.model_api.async_invoke(
            model=self.turbo_model,
            prompt=[{"role": "user", "content": prompt}],
            top_p=0.7, # default 0.7
            temperature=0.9, # default 0.95
        )
        print(response)
        return response

    def embedding_invoke(self, text, top_p=0.7, temperature=0.95):
        response = zhipuai.model_api.invoke(
            model=self.embedding_model,
            prompt=[{"role": "user", "content": text}],
            top_p=0.7, # default 0.7
            temperature=0.9, # default 0.95
        )
        print(response)
        return response

    def embedding_async_invoke(self, prompt, top_p=0.7, temperature=0.95):
        pass

    @staticmethod
    def parse_response(response, type="post"):
        response = {
            "code": 200,
            "data": {
                "choices": [
                    {
                        "content": response
                    }
                ]
            }
        }
        try:
            logger.debug(response)
            if response["code"] != 200:
                logging.error("Error in invoking LLM: " + response["msg"])
                return None
            
            if type == "embedding":
                return response["data"]["choices"][0]["embedding"]
            
            elif type == "post":
                return response["data"]["choices"][0]["content"]
            
            elif type == "react":
                content = response["data"]["choices"][0]["content"]
                if "不感兴趣" in content:
                    return []
                else:
                    reactions = []
                    if "转发" in content:
                        reactions.append("repost")
                    if "关注" in content:
                        reactions.append("follow")
                return reactions

            elif type == "importance":
                content = response["data"]["choices"][0]["content"]
                return int(content)

        except Exception as e:
            logging.error(e)
            return None


class Llama2(LLM):
    def __init__(self, path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(path, device_map='auto')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def invoke(self, prompt):
        #print(model)
        #model = AutoModelForCausalLM.from_pretrained('FlagAlpha/Atom-7B',device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
        #model =model.eval()
        #tokenizer = AutoTokenizer.from_pretrained('FlagAlpha/Atom-7B',use_fast=False)
        prompt = "<s>Human:" + prompt + "</s>\n<s>Assistant:"
        input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
        # input_ids = self.tokenizer([prompt], return_tensors="pt")
        #print(input_ids)
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.3,
            "repetition_penalty": 1.3,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        #print(model)
        generate_ids  = self.model.generate(**generate_input)
        result = self.tokenizer.decode(generate_ids[0])
        return self.parse_response(result)

    def batch_invoke(self, prompts):
        # input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids.to('cuda')
        input_ids = self.tokenizer(prompts, return_tensors="pt")
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.3,
            "repetition_penalty": 1.3,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        generate_ids  = self.model.generate(**generate_input)
        result = self.tokenizer.decode(generate_ids[0])
        print(result)
        return self.parse_response(result)

    def embedding_invoke(self, text) -> np.ndarray:
        input_ids = self.tokenizer(text, return_tensors="pt")
        last_hidden_state = self.model(**input_ids, output_hidden_states=True).hidden_states[-1]
        weights_for_non_padding = input_ids.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

        print(sentence_embeddings.shape)
        return sentence_embeddings[0].detach().numpy()

    def parse_response(self, response):
        return response


class ChatGlm3(LLM):
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path, device_map='auto', trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, device_map='auto', trust_remote_code=True)
        self.model = self.model.eval()

    def invoke(self, text):
        response, history = self.model.chat(self.tokenizer, text, history=[])
        print(response)
        return response
    
    def embedding_invoke(self, text) -> np.ndarray:
        pass

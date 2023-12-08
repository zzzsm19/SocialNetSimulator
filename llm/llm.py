import zhipuai
import logging
zhipuai.api_key = "78a2dfa223061c83018dd3e89b4b09ed.hF7ap5HXcXE8qtMp"

class LLM():
    def __init__(self):
        raise NotImplementedError

    def invoke(self, prompt):
        raise NotImplementedError

    def async_invoke(self, prompt):
        raise NotImplementedError
    
    def embedding_invoke(self, text):
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
            logging.debug(response)
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



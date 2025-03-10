# Copyright 2024 The Chain-of-Table authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import asyncio
import numpy as np

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import giga_config
from src.config import GigaModels, GigaScopes

class GigaChat_CoT:
    def __init__(self):
        self.giga = GigaChat(
            credentials=giga_config.GIGACHAT_TOKEN,
            model=GigaModels.GIGACHAT_PRO,
            scope=GigaScopes.CORP,
            verify_ssl_certs=False
        )

    def get_model_options(
        self,
        temperature=0,
        per_example_max_decode_steps=150,
        per_example_top_p=1,
        n_sample=1,
    ):
        return dict(
            temperature=temperature,
            n=n_sample,
            top_p=per_example_top_p,
            max_tokens=per_example_max_decode_steps,
        )

    def generate_plus_with_score(self, prompt, options=None, end_str=None):
        messages = [
            SystemMessage(content="I will give you some examples, you need to follow the examples and complete the text, and no other content."),
            HumanMessage(content=prompt)
        ]

        giga_response = None
        retry_num = 0
        retry_limit = 2
        error = None

        while giga_response is None:
            try:
                giga_response = self.giga.invoke(messages)
                error = None

            except Exception as e:
                print(str(e), flush=True)
                error = str(e)
                if "This model's maximum context length is" in str(e):
                    print(e, flush=True)
                    giga_response = {
                        "choices": [{"message": {"content": "PLACEHOLDER"}}]
                    }
                elif retry_num > retry_limit:
                    error = "too many retry times"
                    giga_response = {
                        "choices": [{"message": {"content": "PLACEHOLDER"}}]
                    }
                else:
                    time.sleep(60)
                retry_num += 1
        if error:
            raise Exception(error)
        results = []

        text = giga_response.content
        fake_conf = 1
        results.append((text, np.log(fake_conf)))

        return results

    def generate(self, prompt, options=None, end_str=None):
        result =  self.generate_plus_with_score(prompt, options, end_str)
        result = result[0][0]
        return result

async def test_main():
    d = GigaChat_CoT()
    result =  d.generate('Тест связи')
    return result

if __name__ == "__main__":
    res = asyncio.run(test_main())
    print(res)
from openai import OpenAI
import requests
from typing import Callable, Optional
from .response_utils import *
from tqdm import tqdm
import os
import joblib
import time

class LLM:
    """Class implementation of Huggingface inference endpoints LLMs"""
    def __init__(
        self, 
        model: str,
        client: OpenAI,
        gen_params: dict | None = dict()
    ):
        self.model = model
        self.gen_params = gen_params
        self.client = client

    def __call__(
        self, 
        response_type: str,
        prompts: str | list | None = None,
        map_reduce_aggregation: Optional[Callable[[list[str]], str]] = None,
        refine_continue_prompt: str | None = None,
        max_n_refines: int | None = 3,
        map_reduce_agg_max_new_tokens: int | None = None,
        partial_message_queues: Optional[list[list[dict]]] = None
    ):
        supported_response_types = ['single', 'map', 'map_reduce', 'refine', 'reduce']
        assert response_type in supported_response_types, f"Response {response_type} is not supported, please enter one of {supported_response_types}"
        max_new_tokens = self.gen_params.get('MAX_NEW_TOKENS', DEFAULT_MAX_TOKENS)
        message_queue = [
            {'role': 'system', 'content': SYSTEM_PROMPT.format(max_new_tokens=max_new_tokens)}
        ]

        if response_type == 'single' or response_type == 'refine':
            if isinstance(prompts, list):
                assert len(prompts) == 1, "Single/Refine methods should only have one prompt"
                prompt = prompts[0]

            else:
                prompt = prompts

            message_queue = self.LLM_turn(message_queue=message_queue, new_prompt = prompt)
            answer = message_queue[-1]['content'].lstrip('assistant: ')
            
            if response_type == 'refine':
                assert refine_continue_prompt is not None, "Please specify a refine_continue_prompt"
                for _ in tqdm(range(max_n_refines), desc='Refining response...'):
                    message_queue = self.LLM_turn(
                        message_queue = message_queue,
                        new_prompt = refine_continue_prompt
                    )
                    answer = message_queue[-1]['content'].lstrip('assistant: ')

                return LLMRefineResult(result = answer, message_queue = message_queue)
            
            return LLMSingleResult(result = answer, message_queue = message_queue)

        elif response_type == 'map':
            assert isinstance(prompts, list) and len(prompts) > 1, "Please enter a list of >1 prompts for Map methods"
            return self.map(
                prompts = prompts,
                message_queue = message_queue
            )

        elif response_type == 'reduce':
            return self.reduce(
                map_reduce_aggregation = map_reduce_aggregation,
                map_reduce_agg_max_new_tokens = map_reduce_agg_max_new_tokens,
                partial_message_queues = partial_message_queues
            )
        
        elif response_type == 'map_reduce':
            assert isinstance(prompts, list) and len(prompts) > 1, "Please enter a list of >1 prompts for Map-Reduce methods"
            assert map_reduce_aggregation is not None, "Please specify an aggregation function for partial Map-reduce results"
            assert map_reduce_agg_max_new_tokens is not None, "Please specify number of tokens desired in final aggregated Map-Reduce output"

            return self.map_reduce(
                prompts = prompts,
                message_queue = message_queue,
                map_reduce_aggregation = map_reduce_aggregation,
                map_reduce_agg_max_new_tokens = map_reduce_agg_max_new_tokens
            )

    def map(
        self,
        prompts: list[str],
        message_queue: list[dict[str, str]]
    ):
        partial_message_queues = []
        i = 0
        try:
            for prompt in tqdm(prompts, desc='Generating partial results...'):
                new_message_queue = self.LLM_turn(message_queue = message_queue, new_prompt = prompt)
                partial_message_queues.append(new_message_queue)
        except Exception as err:
            print(f"Error during map, skipping prompt")
            i += 1

        return LLMMapResult(partial_message_queues=partial_message_queues)

    def reduce(
        self, 
        map_reduce_aggregation: Callable[[list[str]], str],
        map_reduce_agg_max_new_tokens: int,
        partial_message_queues: list[list[dict]]
    ):
        print('Aggregating partial results...')
        reduced_prompts = map_reduce_aggregation([mq[-1]['content'] for mq in partial_message_queues])
        
        from transformers import AutoTokenizer
        api_token = os.environ['API_TOKEN']

        model_id = self.model
        tokenizer = AutoTokenizer.from_pretrained(model_id, token = api_token)
        
        final_answer = ""
        for idx, prompt in enumerate(reduced_prompts):
            prompt_length = len(tokenizer.tokenize(prompt))
            print(f"Prompt {idx+1} length: {prompt_length}")

            message_queue = [
                {'role': 'system', 'content': SYSTEM_PROMPT.format(max_new_tokens=map_reduce_agg_max_new_tokens)}
            ]

            print(f"Reducing part {idx + 1}...")
            new_message_queue = self.LLM_turn(
                message_queue = message_queue, 
                new_prompt = prompt, 
                map_reduce_agg_max_new_tokens = map_reduce_agg_max_new_tokens
            )
            answer_part = new_message_queue[-1]['content'].lstrip('assistant: ')
            final_answer += answer_part + "\n\n"

        return LLMMapReduceResult(result=final_answer.strip(), partial_message_queues=partial_message_queues)

    def map_reduce(
        self, 
        prompts: list[str],
        message_queue: list[dict[str, str]],
        map_reduce_aggregation: Callable[[list[str]], str],
        map_reduce_agg_max_new_tokens: int
    ):
        partial_message_queues = self.map(
            prompts = prompts,
            message_queue = message_queue
        ).partial_message_queues

        print('Aggregating partial results...')

        message_queue = [
            {'role': 'system', 'content': SYSTEM_PROMPT.format(max_new_tokens=map_reduce_agg_max_new_tokens)},
        ]
        aggregate_prompt = map_reduce_aggregation([mq[-1]['content'].lstrip('assistant: ') for mq in partial_message_queues])
        new_message_queue = self.LLM_turn(
            message_queue=message_queue, 
            new_prompt=aggregate_prompt, 
            map_reduce_agg_max_new_tokens = map_reduce_agg_max_new_tokens
        )
        answer = new_message_queue[-1]['content'].lstrip('assistant: ')

        return LLMMapReduceResult(result=answer, partial_message_queues=partial_message_queues)

    def LLM_turn(
        self, 
        message_queue: list,
        new_prompt: str,
        map_reduce_agg_max_new_tokens: int|None = None
    ):
        tmp = message_queue+[{'role': 'user', 'content': new_prompt}]
        answer = self.answer_message_queue(tmp, map_reduce_agg_max_new_tokens).result
        tmp += [{'role': 'assistant', 'content': answer}]
        return tmp

    def answer_message_queue(
        self,
        message_queue: list,
        map_reduce_agg_max_new_tokens: int | None = None
    ):
        temperature = self.gen_params.get('TEMPERATURE', DEFAULT_TEMPERATURE)
        top_p = self.gen_params.get('TOP_P', DEFAULT_TOP_P)
        stream = self.gen_params.get('STREAM', DEFAULT_STREAM)
        stop_tokens = self.gen_params.get('STOP_TOKENS', DEFAULT_STOP_TOKENS)

        if map_reduce_agg_max_new_tokens:
            max_new_tokens = map_reduce_agg_max_new_tokens
        else:
            max_new_tokens = self.gen_params.get('MAX_NEW_TOKENS', DEFAULT_MAX_TOKENS)

        num_tries = 0
        while num_tries<10: # will try a total of 10 times
            try:
                response = self.client.chat.completions.create(
                    model = self.model,
                    messages = message_queue,
                    temperature = temperature,
                    top_p = top_p,
                    max_tokens = max_new_tokens,
                    stream = stream, 
                    stop = stop_tokens
                )

                answer = ''
                if stream:
                    for chunk in response:
                        if chunk.id: 
                            if chunk.choices[0].delta.content == None and chunk.choices[0].delta.role != None:
                                print(chunk.choices[0].delta.role+': ', end='')
                                answer += chunk.choices[0].delta.role+ ': '
                            elif chunk.choices[0].delta.content != None:
                                print(chunk.choices[0].delta.content, end='')
                                answer += chunk.choices[0].delta.content

                    return LLMResult(result = answer)
                
                else:
                    answer = response.choices[0].message.role + ': ' + response.choices[0].message.content
                    return LLMResult(result = answer)

            except requests.exceptions.HTTPError as err:
                num_tries += 1
                print('Error code: ', err.response.status_code)
                print('Error message: ', err.response.text)
                print('Retrying in 10 seconds')
                time.sleep(10)
            
            except Exception as err:
                num_tries += 1
                print('Error: ', err)
                print('Retrying in 10 seconds')
                time.sleep(10)

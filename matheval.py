import re
import json
import argparse
import jsonlines
from collections import defaultdict
from openai import OpenAI
from typing import Dict, Any
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from latex2sympy2_extended import NormalizationConfig
from transformers import AutoTokenizer
import requests

class MathEvaluator:
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        raise NotImplementedError
    def extract_after_think(self, text: str, truncate_length: int = 1000, finish_generation: bool = True) -> str:
        pattern = r"</think>(.*)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if (match and finish_generation) else text[-truncate_length:]
    
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extracted_answer: str = "", finish_generation: bool = True) -> str:
        raise NotImplementedError
    def llm_judge(self, solution_str: str, ground_truth: str, extracted_answer: str = "", finish_generation: bool = True) -> bool:
        def run_api(inputs):
            request = requests.post(CONSTRUCTED_URL, headers=HEADERS, json=inputs, timeout=10)
            response = request.json()
            return response
        def get_inputs(scene_description):
            body = [
                {"role": "user", "content": scene_description},
            ]

            inputs = {
                "messages": body,
                "max_tokens": 10,
                "stop": "{END}",
                "temperature": 0.0
            }
            return inputs
        
        scene_description = self.get_llm_judge_prompt(solution_str, ground_truth, extracted_answer, finish_generation)
        inputs = get_inputs(scene_description)
        response = run_api(inputs)
        
        return response["choices"][0]["message"]["content"].strip() == "YES"


class AIMEEvaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        # if not ground_truth.startswith("$"):
        #     ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[ExprExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)
            
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()


class GSM8KEvaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        # if not ground_truth.startswith("$"):
        #     ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[ExprExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response with rule-based extracted answer is equivalent to the reference answer from a math question. The final answer may either be enclosed in the \\boxed{{}} or appear after the "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.

1. The reference answer does not include percentage signs, units or time formats (e.g., am, pm), but the Model-generated answer may include them.
For example, 1 is equivalent to 1 %, 1 kg, 1 am, 1 pm, 1:00 am, 1:00 pm, etc.
Model-generated answer: 1%
Reference answer: 1
Your output: YES

Model-generated answer: 1 kg
Reference answer: 1
Your output: YES

Model-generated answer: 1:00 pm
Reference answer: 1
Your output: YES

2. The reference answer only includes one single number, but the Model-generated answer may include multiple numbers.
For example, 10 is equivalent to \\boxed{{(4, 6)}}, etc.
Model-generated answer: 5, 5
Reference answer: 10
Your output: YES

Model-generated answer: 4, 6
Reference answer: 10
Your output: YES

Model-generated answer: 86, 42
Reference answer: 128
Your output: YES

Now let's try a real example.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}
""".strip()


class MATH500Evaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        if not ground_truth.startswith("$"):
            ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[LatexExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()
    
class AMCEvaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        if not ground_truth.startswith("$"):
            ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[LatexExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()


class GPQAEvaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        # if not ground_truth.startswith("$"):
        #     ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[StringExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                StringExtractionConfig(),
            ]
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)
        
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a multiple choice question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()


# class MBPPEvaluator(Evaluator):
#     def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
#         return True, "No extracted answer"
        
#     def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
#         solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
#         return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a multiple choice question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
# Model-generated answer: {solution_str}
# Reference answer: {ground_truth}""".strip()


# class HUMANEVALEvaluator(Evaluator):
#     def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
#         return True, "No extracted answer"
        
#     def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
#         solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
#         return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a multiple choice question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
# Model-generated answer: {solution_str}
# Reference answer: {ground_truth}""".strip()


evaluator_map = {
    "aime2024": AIMEEvaluator(),
    "aime2025": AIMEEvaluator(),
    "gsm8k": GSM8KEvaluator(),
    "math500": MATH500Evaluator(),
    "gpqa_diamond": GPQAEvaluator(),
    "amc23": AMCEvaluator(),
}

API_BASE = None
DEPLOYMENT_NAME = None
API_VERSION = None
CONSTRUCTED_URL = None
API_KEY = None
HEADERS = None

def set_client(api_base=None, deployment_name=None, api_version=None, api_key=None):
    global API_BASE, DEPLOYMENT_NAME, API_VERSION, CONSTRUCTED_URL, API_KEY, HEADERS
    API_BASE = api_base
    DEPLOYMENT_NAME = deployment_name
    API_VERSION = api_version
    CONSTRUCTED_URL = f"{api_base}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    API_KEY = api_key
    HEADERS = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    
    



# def call_llm_judge(message: list, args: argparse.Namespace) -> str:
#     """
#     Call the Qwen API with the given message.
    
#     Args:
#         message (list): Message list for the API.
#         args (argparse.Namespace): Parsed arguments.
        
#     Returns:
#         str: The content of the completion response.
        
#     Raises:
#         Exception: When the API call fails.
#     """

#     try:
#         completion = client.chat.completions.create(
#             model=args.model_name,
#             messages=message,
#             temperature=args.temperature,
#             top_p=args.top_p,
#         )
#         return completion.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"API call failed: {str(e)}")
#         raise


# def rule_judge(completin, ground_truth, dataset_name: str) -> Dict[str, Any]:
#     rule_judge_result = None

#     rule_judge_result = evaluator_map[dataset_name].rule_judge(completin, ground_truth)

#     if not rule_judge_result:
#         print(f"No valid answer detected | LLM judge")
#         call_llm_judge
    
        
#     return example



# def process_example(example: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
#     """
#     Process an individual example with token counting.
#     """
#     try:

#         completion_text = example.get('completion', '')
#         before_truncation_text = completion_text.split("</think>")[0]
#         before_truncation_token_count = count_tokens(before_truncation_text, args)
#         after_truncation_token_count = example.get('generated_tokens', 0) - before_truncation_token_count

#         example['before_truncation_token_count'] = before_truncation_token_count
#         example['after_truncation_token_count'] = after_truncation_token_count

#         example = post_process(example)
#         if not example.get('rule_judge_result', False):
#             think_truncation = extract_after_think(completion_text) or ''

#             prompt = generate_prompt(example, think_truncation)
#             messages = format_prompt(prompt)
#             llm_judge_response = call_qwen(messages, args)
#             print(llm_judge_response)
#             example['llm_judge_response'] = llm_judge_response
#             if llm_judge_response == "YES":
#                 example['llm_judge_result'] = True
#                 example['final_judge_result'] = True
#             elif llm_judge_response == "NO":
#                 example['llm_judge_result'] = False
#                 example['final_judge_result'] = False
#             else:
#                 example['llm_judge_result'] = None
#                 example['final_judge_result'] = None
                
#         return example
#     except Exception as e:
#         print(f"Failed to process example {example.get('idx', 'unknown')}: {str(e)}")
#         example['final_judge_result'] = None
#         return example



# def main():
#     args = parse_arguments()
#     try:
#         final_results = []
#         with jsonlines.open(args.result_save_name, mode='w') as writer:
#             with open(args.data_load_name, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             for idx, example in enumerate(data, start=1):
#                 print(f"Processed example ID: {idx}")
#                 new_example = process_example(example, args)
#                 final_results.append(new_example)

#             writer.write_all(final_results)
#             print(f"Processing complete! Valid results saved to: {args.result_save_name}")

#         grouped_data = defaultdict(list)
#         for entry in final_results:
#             grouped_data[entry['idx']].append(entry)

#         pass_at_1_per_idx = {}
#         for idx, entries in grouped_data.items():
#             correct_count = sum(entry['final_judge_result'] for entry in entries if entry['final_judge_result'] is True)
#             total_count = len(entries)
#             pass_at_1_per_idx[idx] = correct_count / total_count if total_count else 0

#         overall_pass_at_1 = sum(pass_at_1_per_idx.values()) / len(pass_at_1_per_idx)
#         average_generated_tokens = sum(entry['generated_tokens'] for entry in final_results) / len(final_results)
#         average_reasoning_tokens = sum(entry['before_truncation_token_count'] for entry in final_results) / len(final_results)
#         average_answer_tokens = sum(entry['after_truncation_token_count'] for entry in final_results) / len(final_results)

#         print(f'\nPass@1 per idx:\n{pass_at_1_per_idx}\n')
#         print(f'Overall pass@1: {overall_pass_at_1:.4f}')
#         print(f'Average generated_tokens: {average_generated_tokens:.2f}')
#         print(f'Average reasoning_tokens: {average_reasoning_tokens:.2f}')
#         print(f'Average answer_tokens: {average_answer_tokens:.2f}')

#     except json.JSONDecodeError as e:
#         print(f"Data loading failed: Invalid JSON format - {str(e)}")
#         raise


if __name__ == '__main__':
    set_client(api_base="", deployment_name="", api_version="", api_key="")
    print("123")
    print(evaluator_map["aime2024"].llm_judge("1", "1", "1", True))

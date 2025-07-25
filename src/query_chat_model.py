import json
from math import ceil
import random
import sys
import time
from datetime import datetime
from openai import OpenAI
import openai
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential, get_bearer_token_provider
import config
from config import debug_print
from static_mutation import prune_equivalent_codes
from assertion_rewriter import rewrite_assert
from typing import Dict, Any, List, Tuple
import json
from dataclasses import dataclass
from enum import Enum
import os
import subprocess

# System prompts
HASKELL_TO_PYTHON_LLM_PROMPT = """You are a strict Haskell-to-Python translator operating within a formal DSL synthesis pipeline.
Input: A pure, safe Haskell function (no recursion, no IO, no imports).
Output: A single valid Python function that strictly follows these constraints:
Use only:
map, filter, reduce,
list comprehensions,
pure expressions and basic conditionals,
Do not use:
imports,
mutation or reassignment,
recursion,
IO,
advanced syntax or any libraries,
The return value must match Haskell semantics exactly
Output only Python code. No markdown, no explanations, no comments.
If provided, treat {bad_examples_section} as known anti-patterns to avoid.
"""

HASKELL_SYSTEM_PROMPT_TEMPLATE = """You are a strict Haskell code generator operating in a formal DSL synthesis pipeline.
Your task is to generate a pure, stateless Haskell function using only canonical functional primitives:
- `map`, `filter`, `fold`, list comprehensions, and simple `let` bindings
- No recursion, pattern matching, IO, mutation, or side effects
- Functions must be total, deterministic, and structurally simple
- Do not use any imports or libraries
{bad_examples_section}
Only output the Haskell code block. Do not include comments or explanation.

"""

HASKELL_JUDGE_PROMPT = """You are a code compliance judge for Haskell functions in a DSL pipeline. Determine whether the function uses only:
- map, filter, fold, list comprehensions, let-bindings
Forbidden: recursion, pattern matching, IO, imports, or mutation.
Respond with:
- ✅ Haskell-Compatible or ❌ Rejected
- Short reason and code excerpts if rejected
"""

PYTHON_JUDGE_PROMPT = """
You are a code compliance judge in a DSL synthesis pipeline. You are given a Python function and a set of test cases. Your task is to evaluate whether the function:

1. ✅ Is syntactically valid Python (no syntax errors, malformed formatting, or non-Python syntax)
2. ✅ Is properly formatted (no markdown, no backticks, no Haskell-style or pseudo-code)
3. ✅ Complies with the DSL rules
4. ✅ Passes the provided test cases (or is at least *logically aligned* with the test case expectations)

DSL Rules:
- ✅ Allowed: map, filter, reduce, list comprehensions, simple for-loops, arithmetic, and conditionals
- ❌ Forbidden: any import, IO operations, use of `re`, `json`, `open`, `eval`, mutation of external state, recursion, or advanced Python syntax

Test Format:
The test cases are written using standard `assert` statements that call the target function and expect specific outputs.

Your Response:
- ✅ DSL-Compatible if the code is syntactically valid, DSL-compliant, and will likely pass all test cases
- ❌ Rejected if it fails any of the above checks

If ❌ Rejected:
- Brief explanation of why
- Code excerpt(s) or test(s) that fail

Tests:
"""



def gen_and_prune_codes(client, prog_data, tests_in_ctxt, token_counter=None):
    if not config.use_oracle_as_code_suggestion:
        orig_codes = get_code_suggestions(client,
            prog_data, tests_in_ctxt, token_counter=token_counter)
    else:
        orig_codes = [prog_data['oracle']]

    print(f"Finished generating {len(orig_codes)} code suggestions")

    # prune code that contains a user input (TODO: use ASTs to prune)
    codes = [code for code in orig_codes if 'input(' not in code]

    # prune equivalent codes
    codes = prune_equivalent_codes(codes)

    print(
        f"Retained {len(codes)} code suggestions after removing equivalent codes")
    return orig_codes, codes

def get_code_suggestions(client, prog_data, tests_in_ctxt, token_counter):
    # take a context and a signature
    # create a prompt for Codex
    #prompt = get_prompt(prog_data)
    #f tests_in_ctxt is not None:
        # add " " as the code extraction eats away a space (see get_codex_code_suggestions)
        #prompt = " " + "\n".join(tests_in_ctxt) + prompt

    #print("Prompt for Code Generation:")
    #print('-' * 80)
    #print(prompt)
    #print('-' * 80)
    
    additional_tests = " " + "\n".join(tests_in_ctxt)
    # query codex for suggestions
    code_suggestions = get_custom_code_suggestions(client, prog_data, additional_tests, config.MAX_NUM_CODEX_CODE_SUGGESTIONS)
    
    debug_print('Codes' + '*' * 80)
    for suggestion in code_suggestions:
        debug_print(suggestion)
        debug_print('-' * 80)
    debug_print('*' * 80)
    debug_print(f"number of code suggestions {len(code_suggestions)}")
    if len(code_suggestions) == 0:
        return []

    # prune the suggestions
    # code_suggestions = prune_codes_that_dont_pass_wff_check(code_suggestions)
    debug_print(
        f"number of code suggestions after synactic pruning {len(code_suggestions)}")
    debug_print("$" * 80)
    if (len(code_suggestions) == 0):
        return []

    return code_suggestions


def get_test_suggestions(client, prog_data, num_tests, code_suggestions, token_counter=None):
    """Generate tests using testgen"""
    # generate tests
    test_suggestions = get_codex_test_suggestions(client,
        prog_data, num_tests, code_suggestions, token_counter=token_counter) if num_tests > 0 else []
    # prune equivalent codes
    test_suggestions = prune_equivalent_codes(test_suggestions)
    debug_print('Generated Tests' + '*' * 80)
    for suggestion in test_suggestions:
        debug_print(suggestion)
        debug_print('-' * 80)
    debug_print('*' * 80)
    print(
        f"Finished generating {len(test_suggestions)} number of test suggestions ")
    # if user_fixes_tests flag, then rewrite each test using assertion rewriter
    if False: # config.user_fixes_tests:
        # only use user_fixes_tests with single_assert with a message
        if not config.single_assert_per_test:
            raise Exception(
                "user_fixes_tests only works with single_assert_per_test")
        fixed_tests = []
        for test in test_suggestions:
            try:
                fixed_tests.append(rewrite_assert(prog_data['oracle'], test))
                debug_print(f"rewrote {test} to {fixed_tests[-1]}")
            except:
                import traceback
                debug_print(f"Exception in rewriter: {traceback.format_exc()}")
                continue
        test_suggestions = fixed_tests
        test_suggestions = prune_equivalent_codes(test_suggestions)
    return test_suggestions


def filter_response(resp_text):
    if "```python\n" in resp_text:
        resp_text = resp_text.split("```python\n")[1]
        if "```" in resp_text:
            resp_text = resp_text.split("```")[0]
    elif "```\n" in resp_text:
        resp_text = resp_text.split("```")[1]
        if "```" in resp_text:
            resp_text = resp_text.split("```")[0]
    if "<code>" in resp_text:
        resp_text = resp_text.split("<code>")[1]
        if "</code>" in resp_text:
            resp_text = resp_text.split("</code>")[0]
    if "`\n" in resp_text:
        resp_text = resp_text.split("`")[1]
        if "`" in resp_text:
            resp_text = resp_text.split("`")[0]
    return resp_text

def get_prompt(prog_data):
    """Get a prompt for ChatCompletion API"""
    context = prog_data['ctxt']
    function_signature = prog_data['sig']

    prompt_text = f"Complete the following Python function:\n\n{function_signature}\n\n"
    if context.strip() != "":
        prompt_text += f"The context of the function is :\n\n{context}\n\n"
    prompt_text += "Surround the function with <code> and </code> tags.\n"
    prompt_text += "Do not explain the function, just complete the function.\n"
    prompt = [
        {
            "role": "system",
            "content": "Suppose you are a code completion engine. You are asked to complete the following Python function. " +
            "The function signature is given below. The context of the function is also provided. Complete the function. "
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]
    return prompt

def openai_chat_completion(client, messages: List[Dict[str, str]], model: str = "gpt-4o", temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        n=1
    )
    return response


def format_bad_examples(examples: List[str]) -> str:
    if not examples:
        return ""
    return "\nAvoid the following invalid examples:\n" + "\n".join(f"\n{ex}\n```" for ex in examples)
    
def generate_valid_code(
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
    token_to_check: str,
    judge_prompt: str,
    max_iterations: int = 10,
    model: str = "gpt-4o",
    coder_temp: float = 0.8,
    judge_temp: float = 0,
    max_tokens: int = 1024,
    filter: bool = True) -> Tuple[bool, str, str]:
    """
    Run LLM call repeatedly:
    - Generate code
    - Judge the code for presence of token

    Returns:
        (bool, str, str):
            bool - True if exited because token disappeared
            str  - Last code generated
            str  - Last judgement result
    """
    bad_code = []
    
    for _ in range(max_iterations):
        formatted_system_prompt = system_prompt.format(
            bad_examples_section=format_bad_examples(bad_code)
        )

        # Generate code
        gen_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=coder_temp,
            n=1
        )
        generated_code = gen_response.choices[0].message.content.strip()
        if filter:
            clean_code = filter_response(generated_code)
        else:
            clean_code = generated_code
            
        bad_code.append(clean_code)

        # Judge the generated code
        judge_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": clean_code},
            ],
            max_tokens=max_tokens,
            temperature=judge_temp,
            n=1
        )
        judgement = judge_response.choices[0].message.content.strip()
            
        if token_to_check not in judgement:
            return True, clean_code, judgement

    return False, clean_code, judgement

def run_openai_pipeline(client, function_description: str,additional_tests) -> Dict[str, str]:
    
    # Generate Haskell
    hs_valid, haskell_code, hs_judgement = generate_valid_code(client,
                                                    system_prompt = HASKELL_SYSTEM_PROMPT_TEMPLATE,
                                                    user_prompt = function_description,
                                                    token_to_check = "❌",
                                                    judge_prompt = HASKELL_JUDGE_PROMPT)


    # Generate Python
    py_valid, python_code, py_judgement = generate_valid_code(client,
                                                    system_prompt = HASKELL_TO_PYTHON_LLM_PROMPT,
                                                    user_prompt = function_description,
                                                    token_to_check = "❌",
                                                    judge_prompt = PYTHON_JUDGE_PROMPT+"\n"+additional_tests)
    print(f"Haskell:\n{haskell_code}")
    print(f"Judgment: {hs_valid,hs_judgement}")
    print(f"Python:\n{python_code}")
    print(f"Judgment: {(py_valid,py_judgement)}")
    return {
        "haskell": haskell_code,
        "haskell_judge": (hs_valid,hs_judgement),
        "python": python_code,
        "python_judge": (py_valid,py_judgement)
    }
        
def get_custom_code_suggestions(client,prog_data,additional_tests,n_iterations) -> List[str]:
    function_description = prog_data['sig'] + "\n\n" + prog_data['ctxt']
    print("HIYAAAA")
    print(function_description)
    results_dict = run_openai_pipeline(client, function_description,additional_tests)
    return [results_dict["python"] for _ in range(n_iterations)]


def get_codex_code_suggestions(client, context, prompt, num_sugg, token_counter):
    """Get code suggestions"""
    response, response_len = get_codex_response_with_retries(client,
        prompt, num_sugg, config.sampling_temperature, False, num_sugg, token_counter=token_counter
    )
    generations = []
    for index in range(response_len):
        code = response.choices[index].message.content
        code = filter_response(code)
        generations.append(
            context + "\n" + code.strip() + "\n"
        )
    return generations


def extract_code_from_codex_suggestion(code):
    """Extract the python code from the model suggestion"""
    new_code = code.replace('\r\n', '\n')
    lines = new_code.split('\n')
    # find the index of the line that is empty
    indices = [i for i, line in enumerate(lines) if len(line) == 0]
    index = indices[0] if len(indices) > 0 else len(lines)
    returnval = '\n'.join(lines[:index])
    # debug_print(f"extracted code from \n'''{new_code}'''\n is \n'''{returnval}'''\n")
    return returnval


def get_codex_test_suggestions(client, prog_data, num_tests, code_suggestions, token_counter):
    """Get test suggestions from model"""
    debug_print(f"sig = {prog_data['sig']}")

    # to check the metrics for validation set, simply check how many examples have at least 1 True in the code predicted
    # This can be done by grepping the output file

    # get the prompt
    debug_print(f"test_gen_option = {config.test_gen_option}")
    prompt = mk_test_suggestion_prompt(
        prog_data, f"{prog_data['sig']}\n\tpass")

    # consider the validation tests as the prompt (only in query_oracle model)
    if config.test_gen_option == 'ideal':
        assert config.query_oracle_opt == True
        return prog_data['val_tests']
    elif config.test_gen_option == 'skip':  # take the context and ask for test suggestions
        prompt = mk_test_suggestion_prompt(prog_data, f"{prog_data['sig']}")
    elif config.test_gen_option == 'pass':
        prompt = mk_test_suggestion_prompt(
            prog_data, f"{prog_data['sig']}\n\tpass")
    elif config.test_gen_option == 'top1':  # take the top code suggestion as prompt
        # assert len(code_suggestions) > 0, f"Fails as o code suggestions for {sig} in top1 mode"
        # the above assert may fail if the prompt cannot fit into LM buffer
        prompt = mk_test_suggestion_prompt(prog_data, code_suggestions[0] if len(
            code_suggestions) > 0 else f"{prog_data['sig']}\n\tpass")
    elif config.test_gen_option == 'random':  # take a random code suggestion as prompt
        prompt = mk_test_suggestion_prompt(prog_data, random.choice(
            code_suggestions) if len(code_suggestions) > 0 else f"{prog_data['sig']}\n\tpass")
    elif config.test_gen_option == 'oracle':  # this is only for evaluation, not at actual usage
        prompt = mk_test_suggestion_prompt(prog_data, prog_data['oracle'])
    elif config.test_gen_option == 'suggestions':
        tests_w_code = []
        # generate min(|code_suggestions|, num_tests) tests
        for c in code_suggestions:
            prompt = mk_test_suggestion_prompt(prog_data, c)
            tests_w_code.append((c, get_codex_test_suggestions_from_prompt(
                prompt, 1, prog_data['func_name'])[0]))
            if len(tests_w_code) >= num_tests:
                break
        return [x[1] for x in tests_w_code]  # return only the tests
    else:
        raise Exception(f"Invalid test_gen_option {config.test_gen_option}")

    return get_codex_test_suggestions_from_prompt(client, prompt, num_tests, prog_data['func_name'], token_counter=token_counter)


def get_codex_test_suggestions_from_prompt(client, prompt, num_tests, func_name, token_counter):
    print("Prompt for Test Generation:")
    print('-' * 80)
    print(prompt)
    print('-' * 80)
    response, response_len = get_codex_response_with_retries(client,
        prompt, num_tests, config.sampling_temperature, False, num_tests, token_counter=token_counter)

    """return [
        # '\ndef ' + TEST_PREFIX + func_name + '():\n' + extract_code_from_codex_suggestion(response['choices'][index]['text'])
        test_sig + prompt.split(test_sig)[-1] + extract_code_from_codex_suggestion(
            response['choices'][index]['text'])
        for index in range(response_len)
    ]"""
    generations = []
    for index in range(response_len):
        test = response.choices[index].message.content
        test = filter_response(test.strip())
        generations.append(test)
    return generations


def mk_test_suggestion_prompt(prog_data, code_suggestion):
    """Make a test suggestion prompt"""

    # prompt_text = f"{prog_data['ctxt']}\n{code_suggestion}\n\ndef {config.TEST_PREFIX}{prog_data['func_name']}():\n\tassert {prog_data['func_name']}("
    # debug_print(f"test_prompt = {prompt}")
    prompt = [
        {
            "role": "system",
            "content": "Suppose you are a code completion engine. You are asked to generate tests for a Python function. \n" +
            "You will be given a function which contains the description. \n" +
            "You need to generate tests for the function. "
        }
    ]
    prompt_text = f"Context of the function is :\n\n{prog_data['ctxt']}\n\n"
    prompt_text += f"The functions is defined as follows:\n\n{code_suggestion}\n\n"
    prompt_text += f"Generate a test code for the function containing assersions. \n"
    prompt_text += f"Start the test code with: \n\ndef {config.TEST_PREFIX}{prog_data['func_name']}():\n\tassert {prog_data['func_name']} (\n\n\n"
    prompt_text += "Surround the test code with <code> and </code> tags.\n"
    prompt_text += f"Do not explain the test code, just generate it. Do not call the test code.\n"
    prompt_text += f"Do not write any standalone asserts.\n"
    prompt_text += "The test code should contain only one assertion for the function. \n"

    prompt.append(
        {
            "role": "user",
            "content": prompt_text
        }
    )
    return prompt


def get_codex_response_with_retries(client, prompt_val, best_of_val, temperate_val, echo_val, num_sugg, token_counter):
    """Query codex with retries"""
    sleep_time = 5  # seconds
    for _ in range(config.NUM_CODEX_RETRIES):
        try:
            response = get_or_create_codex_response(client,
                prompt_val, best_of_val, temperate_val, echo_val,
                config.MAX_TOKENS, config.MODEL, num_sugg, token_counter
            )
            response_len = min(len(response.choices), num_sugg)
            return response, response_len
        except openai.RateLimitError as e:
            import traceback
            traceback.print_exc()
            print(f"Exception in get_codex_response_with_retries {e}")
            time.sleep(sleep_time)
            sleep_time *= 2
            continue
        except openai.APIConnectionError as e:
            import traceback
            traceback.print_exc()
            print(f"Exception in get_codex_response_with_retries {e}")
            time.sleep(5)
            continue
    return None


def get_or_create_codex_response(client, prompt_val, best_of_val, temp_val, echo_val, max_tokens_val, engine, num_sugg, token_counter: config.TokenCounter):
    # https://beta.openai.com/docs/api-reference/completions/create
    # {prompt, max_tokens, temperature, top_p, n, stream, logprobs, echo, stop, presence_penalty, frequency_penalty, best_of, logit_bias, engine}
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    max_suggestions = max(config.MAX_NUM_CODEX_CODE_SUGGESTIONS,
                          config.MAX_NUM_CODEX_TEST_SUGGESTIONS)
    # the cache should not be sensitive to the number of suggestions
    # however, this would be too expensive when calling test per code (typically 50 code suggestion and 100 test suggestions per code, 5000 queries per example !!)
    k = (prompt_val, max_suggestions, temp_val, echo_val,
         max_tokens_val, engine, max_suggestions)

    if config.codex_cache_file is not None and str(k) in config.codex_query_response_log:
        config.skip_codex_query_cnt = config.skip_codex_query_cnt + 1
        resp = config.codex_query_response_log[str(k)][1]
        debug_print(f"Cached response for {k} is {resp}")
        return resp
    assert best_of_val <= max_suggestions
    assert num_sugg <= max_suggestions
    config.mk_codex_query_cnt = config.mk_codex_query_cnt + 1
    num_prompt_tokens = config.count_tokens(prompt_val, config.MODEL)
    response = None
    if num_prompt_tokens + (max_suggestions * max_tokens_val) > token_counter.token_limit:
        allowed_suggestions = (token_counter.token_limit - num_prompt_tokens) // max_tokens_val
        number_of_trials = ceil(max_suggestions / allowed_suggestions)
    else:
        number_of_trials = 1
        allowed_suggestions = max_suggestions
    print("Tokens in prompt: ", num_prompt_tokens, "\tMax suggestions : ", max_suggestions, "\tMax tokens per suggestion: ", max_tokens_val)
    print(
        f"Query will take {number_of_trials} trials with {allowed_suggestions} suggestions each" +\
        f" to generate {max_suggestions} suggestions, with {max_suggestions - (number_of_trials - 1) * allowed_suggestions} suggestions in the last trial."
    )
    while response is None or len(response.choices) < max_suggestions:
        if response is not None:
            allowed_suggestions = min(allowed_suggestions, max_suggestions - len(response.choices))
        while token_counter.is_over_limit(num_prompt_tokens + allowed_suggestions * max_tokens_val):
            print(f"Need {num_prompt_tokens + allowed_suggestions * max_tokens_val} token quota. Sleeping 5 seconds for token counter reset.")
            time.sleep(5)
        try:
            query_response = client.chat.completions.create(
                model=config.MODEL,
                messages=prompt_val,
                max_tokens=max_tokens_val,
                n=allowed_suggestions,
                temperature=temp_val,
            )
        except openai.RateLimitError as e:
            time_to_sleep = 60
            if "Please retry after" in str(e):
                try:
                    time_to_sleep = int(str(e).split("Please retry after")[1].split("seconds")[0].strip())
                except:
                    pass
            print(f"Rate limit error, sleeping for {time_to_sleep} seconds")
            time.sleep(time_to_sleep)
            continue
        if response is None:
            response = query_response
        else:
            response.choices += query_response.choices
        
        token_counter.add_tokens(query_response.usage.total_tokens)
        print("Current Tokens:, ", query_response.usage.total_tokens, "\tUsed tokens: ",
            token_counter.used_tokens, "\tToken limit: ", token_counter.token_limit,
            "\tSo far generated: ", len(response.choices))
    v = (k, response, current_time)
    config.codex_query_response_log[str(k)] = v
    return response



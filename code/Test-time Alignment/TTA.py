from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch.nn import functional as F
import torch
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os

def get_input_ids(tokenizer, dev, prompt: str, template: str = 'it', add_generation_prompt: bool = False) -> torch.Tensor:
    if template == 'it':
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    elif template == 'chat':
        tokens = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt"
        ).to(dev)
    else:
        raise ValueError(f"Unknown template: {template}")
    return tokens


def tokens_to_text(tokenizer, tokens: torch.Tensor, skip_special_tokens=True) -> List[str]:
    
    return tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)
 
def is_end_token(token: str, end_map: List[str]) -> bool:
    return token in end_map


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def calculate_token_score_with_prepending(
    new_token: torch.Tensor,
    prefix: torch.Tensor,
    dpo_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
) -> float:

    
    combined_input = torch.cat([prefix, new_token], dim=-1)
    decoded_combined_input = llm_tokenizer.decode(combined_input[0])


    tokenized_combined_input = dpo_tokenizer.encode(decoded_combined_input, add_special_tokens=False, return_tensors="pt")

    decoded_token = llm_tokenizer.decode(new_token[0]) 
    tokenized_input = dpo_tokenizer.encode(decoded_token, add_special_tokens=False, return_tensors="pt").to(prefix.device)
    
    num_sub_tokens = tokenized_input.size(-1)
    
    
    with torch.no_grad():
        dpo_outputs = dpo_model(input_ids=tokenized_combined_input.to(dpo_dev))
        ref_outputs = ref_model(input_ids=tokenized_combined_input.to(ref_dev))
    

    if num_sub_tokens == 1:
        dpo_logits = dpo_outputs.logits[:, -num_sub_tokens - 1:-1, :].to(llm_dev)
        ref_logits = ref_outputs.logits[:, -num_sub_tokens - 1:-1, :].to(llm_dev)
    else:
        dpo_logits_slice = dpo_outputs.logits[:, -num_sub_tokens - 1:-1, :].to(llm_dev)
        ref_logits_slice = ref_outputs.logits[:, -num_sub_tokens - 1:-1, :].to(llm_dev)
        
        dpo_logits = torch.mean(dpo_logits_slice, dim=1, keepdim=True)  # Shape: [batch_size, 1, n]
        ref_logits = torch.mean(ref_logits_slice, dim=1, keepdim=True)    # Shape: [batch_size, 1, n]

    

    token_indices = tokenized_input[0, -1:]

    dpo_probs = F.softmax(dpo_logits, dim=-1).gather(dim=-1, index=token_indices.unsqueeze(-1).unsqueeze(-1)).squeeze(-1)
    ref_probs = F.softmax(ref_logits, dim=-1).gather(dim=-1, index=token_indices.unsqueeze(-1).unsqueeze(-1)).squeeze(-1)


    dpo_probs = torch.clamp(dpo_probs, min=1e-4)
    ref_probs = torch.clamp(ref_probs, min=1e-4)
    reward = torch.log(dpo_probs).mean() - torch.log(ref_probs).mean()
    

    return reward.item()

def find(text):
    # target_substr = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    target_substr = '<|im_start|>assistant'

    index = text.find(target_substr)

    if index != -1:
        content_after_target = text[index + len(target_substr):]
        return content_after_target
    else:
        return None

def greedy_decoding(
    input_ids: torch.Tensor,
    max_length: int,
    candidate_width: int,
    alpha: float,
    beta: float,
    llm: AutoModelForCausalLM,
    dpo_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,                   
    use_rank: bool,
    debug: bool
) -> str:

    input_ids = input_ids.unsqueeze(0)

    beams = [(input_ids, 0, 0)]

    
    for _ in range(max_length):
        all_candidates = []

        beam_tokens, beam_score, length = beams[0]

        with torch.no_grad():
            outputs = llm(input_ids=beam_tokens)
        logits = outputs.logits[:, -1, :]
        topk_logits, topk_indices = torch.topk(logits, candidate_width, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        

        reward_dict = {}
        new_tokens = [topk_indices[:, idx].unsqueeze(0) for idx in range(candidate_width)]
        rewards = [calculate_token_score_with_prepending(new_tokens[idx], beam_tokens, dpo_model, ref_model) for idx in range(candidate_width)]
        rewards = softmax(rewards)

        token_reward_pairs = list(zip(new_tokens, rewards))
        token_reward_pairs.sort(key=lambda x: x[1], reverse=True)
        for rank, (token, token_reward) in enumerate(token_reward_pairs, start=1):
            reward_dict[token.item()] = (token_reward, rank)

        for idx in range(candidate_width):
            new_token = topk_indices[:, idx].unsqueeze(0)
            if debug: print('candiate: ', tokenizer.decode(new_token[0]))
            if torch.is_tensor(beam_score):
                beam_score = beam_score.cpu().item()
                
            if use_rank:
                new_score = beam_score + alpha * (-idx)
            else:
                if torch.is_tensor(topk_probs[:, idx][0]):
                    token_prob = topk_probs[:, idx][0].cpu().item()
                else:
                    token_prob = topk_probs[:, idx][0]
                new_score = alpha * token_prob 
            if use_rank:
                token_reward = - reward_dict[new_token.item()][1]
            else:
                token_reward = reward_dict[new_token.item()][0]

            total_score = new_score + beta * token_reward

            normalized_score = total_score / (length + 1)

            updated_candidate = (torch.cat([beam_tokens, new_token], dim=1), normalized_score, length + 1)
            decoded_string = tokenizer.decode(updated_candidate[0][0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

            decoded_string_find = find(decoded_string)[1:]
            if idx < 1 and any(end_token in decoded_string_find for end_token in END_MAP):
                return decoded_string

            all_candidates.append(updated_candidate)


        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)

        beams = [all_candidates[0]]

        

def remove_end_tokens(s, end_map):
    for token in end_map:
        s = s.replace(token, '')
    return s



if len(sys.argv) != 13:
    sys.exit(1)


mode = sys.argv[1]
alpha = float(sys.argv[2])
beta = float(sys.argv[3])
lg = sys.argv[4]
cuda_1 = sys.argv[5]
cuda_2 = sys.argv[6]
cuda_3 = sys.argv[7]
cuda_4 = sys.argv[8]
version = sys.argv[9]
llm_path = sys.argv[10]
dpo_path = sys.argv[11]
ref_path = sys.argv[12]

ref = ""
if "Qwen" in ref_path:
    ref = "qwen_reward"
elif "llama" in ref_path:
    ref = "llama_reward"

LLM_PATH = llm_path
DPO_PATH = dpo_path
REF_PATH = ref_path

llm_dev_1 = f"cuda:{cuda_1}"
llm_dev_2 = f"cuda:{cuda_2}"
llm_dev = llm_dev_1
dpo_dev = f"cuda:{cuda_3}"
ref_dev = f"cuda:{cuda_4}"
torch_dtype = torch.float16



llm = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch_dtype,
    device_map='auto'
).eval()

dpo_model = AutoModelForCausalLM.from_pretrained(DPO_PATH, torch_dtype=torch_dtype).to(dpo_dev).eval()
ref_model = AutoModelForCausalLM.from_pretrained(REF_PATH, torch_dtype=torch_dtype).to(ref_dev).eval()

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
dpo_tokenizer = AutoTokenizer.from_pretrained(DPO_PATH)

END_MAP = [
    '\n', 
    '.\n\n', 
    '!\n\n', 
    '?\n\n', 
    '。\n\n', 
    '...\n\n', 
    ' \n\n', 
    '\n\n', 
    llm_tokenizer.eos_token,
    '<|im_end|>',
]


with open(f'mt-decoding/results/{lg}.txt', 'r') as file:
    source = [line.strip() for line in file]

end = 500

template = 'chat'


output = f'mt-decoding/results/qwen/output_{lg}_{mode}_{version}_{alpha}_{beta}_{end}_{ref}_500.txt'
if os.path.exists(output):
    os.remove(output)

result = lg.split('-')
lang1 = result[0]
lang2 = result[1]

lang_dict = {"zh":"Chinese", "en":"English", "de":"German", "ru":"Russian"}

for sr in source:
    prompt = [{"role": "system", "content": "You are a helpful machine translation assistant."}, {"role": "user", "content": f"Translate the following {lang_dict[lang1]} source text to {lang_dict[lang2]} without any other text. \n{lang_dict[lang1]}: {sr}\n{lang_dict[lang2]}:"}]



    input_ids = get_input_ids(tokenizer=llm_tokenizer, dev=llm_dev,prompt=prompt, template = template, add_generation_prompt=True)  
    

    output_text = greedy_decoding(input_ids[0], max_length=500, candidate_width=5, alpha=alpha, beta=beta, llm=llm, dpo_model=dpo_model, ref_model=ref_model, tokenizer=llm_tokenizer, use_rank=False, debug=False)

    try:
        ans = remove_end_tokens(find(output_text)[1:], END_MAP)
    except:
        ans = ""

    with open(output, 'a') as file2:
        file2.write(ans +'\n')

# Example Command:
# CUDA_VISIBLE_DEVICES=0,3 python TTA.py "GD" 1 0.5 "zh-en" 0 0 0 1 "ours" "Qwen/Qwen2.5-14B-Instruct" "model/Llama3.2-3B-Instruct/dpo/full/epoch2" "meta-llama/Llama-3.2-3B-Instruct"
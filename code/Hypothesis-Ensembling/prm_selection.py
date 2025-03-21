import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

import argparse
import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 消除warning
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MT:
    def __init__(self, torch_dtype=torch.float16):
        #DPO_PATH = '/mnt/zju105100171/home/jiahan/model/Llama3.2-3B-Instruct/dpo/full/epoch2'
        #REF_PATH = '/mnt/zju105100171/home/jiahan/model/Llama3.2-3B-Instruct/base/Llama-3.2-3B-Instruct'
        DPO_PATH = '/mnt/zju105100171/home/jiahan/model/Qwen2.5-3B-Instruct/dpo/full/epoch2'
        REF_PATH = '/mnt/zju105100171/home/jiahan/model/Qwen2.5-3B-Instruct/base/Qwen2.5-3B-Instruct'
        self.dpo_device = 'cuda:2'
        self.ref_device = 'cuda:2'  # 好像能够放到一张卡上
        self.language_name = [['German','English'],['English','German'],['English','Russian'],['Rassian','English'],['Chinese','English'],['English','Chinese']]
        self.dpo_LLM = AutoModelForCausalLM.from_pretrained(DPO_PATH, torch_dtype=torch.bfloat16).to(self.dpo_device).eval()
        self.ref_LLM = AutoModelForCausalLM.from_pretrained(REF_PATH, torch_dtype=torch.bfloat16).to(self.ref_device).eval()
        self.llm_tokenizer= AutoTokenizer.from_pretrained(REF_PATH)
        self.debug = False
    
    def tokens_to_text(self, tokenizer, tokens: torch.Tensor, skip_special_tokens=True):
        return tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def get_input_ids(self,tokenizer, prompt:str, template:str = 'it',add_generation_prompt: bool = False) -> torch.Tensor:
        if template == 'it':
            tokens = tokenizer(prompt, return_tensors="pt").input_ids
        elif template == 'chat':
            tokenizer.pad_token = tokenizer.eos_token
            tokens = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt"
            )
        else:
            raise ValueError(f"Unknown template: {template}")
        return tokens
    
    def score(self, language_name: list, src: str, mt: str, method: str):
        mt_prompt = [{"role": "user", "content": 'Translate the following text from ' + language_name[0] + ' into ' + language_name[1] + '.\n' + language_name[0] + ': ' + src + '\n' + language_name[1] + ':'}, {'role':'assistant', 'content': mt}]
        prompt = [{"role": "user", "content": 'Translate the following text from ' + language_name[0] + ' into ' + language_name[1] + '.\n' + language_name[0] + ': ' + src + '\n' + language_name[1] + ':'}]
        mt_token_list = self.get_input_ids(tokenizer=self.llm_tokenizer,prompt=mt_prompt,template='chat',add_generation_prompt=True).to(self.dpo_device)
        prompt_token_list = self.get_input_ids(tokenizer=self.llm_tokenizer,prompt=prompt,template='chat',add_generation_prompt=True).to(self.dpo_device)
        length = len(mt_token_list[0]) - len(prompt_token_list[0])
        prompt_length = len(prompt_token_list[0])
        reward = 0

        if self.debug: 
            print('mt_token_list')
            print(mt_token_list)

        for i in range(length):
            if self.debug: 
                print(prompt_token_list)
                print(mt_token_list[0][i])

            prompt_token_list = prompt_token_list.to(self.dpo_device)
            attention_mask = torch.ones(prompt_token_list.shape,dtype=torch.int8).to(self.dpo_device)
            dpo_results = self.dpo_LLM.generate(     
                prompt_token_list,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                pad_token_id=self.llm_tokenizer.eos_token_id
                )   
            dpo_results = torch.softmax(dpo_results.scores[0],dim = 1)
            dpo_probability = dpo_results[0][mt_token_list[0][prompt_length + i]]

            attention_mask = torch.ones(prompt_token_list.shape,dtype=torch.int8).to(self.ref_device)
            prompt_token_list = prompt_token_list.to(self.ref_device)
            ref_results = self.ref_LLM.generate(
                prompt_token_list,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                pad_token_id=self.llm_tokenizer.eos_token_id
                )
            ref_results = torch.softmax(ref_results.scores[0],dim = 1)
            ref_probability = ref_results[0][mt_token_list[0][prompt_length + i]].to(self.dpo_device)
            
            if method == 'weighted':
                reward += torch.log(dpo_probability / ref_probability) / (i + 1)
            elif method == 'length normalized':
                reward += torch.log(dpo_probability / ref_probability) / length
            prompt_token_list = torch.cat((prompt_token_list,mt_token_list[0][prompt_length + i].unsqueeze(0).unsqueeze(0)),dim=1)
        return reward


def read_input(file_path):
    """
    读取输入文件，提取每例的 source 和 hypotheses。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    examples = []
    for i in range(0, len(lines), 9):  # 每 9 行为一例
        source = lines[i].strip()  # 第一行为 source
        hypotheses = [line.strip() for line in lines[i + 1 : i + 9]]  # 后 8 行为 hypotheses
        examples.append({"source": source, "hypotheses": hypotheses})
    return examples

def get_language_pair(input_file_name):
    """
    从输入文件名中提取语言对（如 zh-en 或 en-zh）。
    """
    if "zh-en" in input_file_name:
        return "zh-en"
    elif "en-zh" in input_file_name:
        return "en-zh"
    else:
        raise ValueError("输入文件名中未找到有效的语言对（zh-en 或 en-zh）。")

def get_output_file_name(input_file_name, model):
    """
    根据输入文件名、模型名称和当前日期生成输出文件名。
    格式：{模型}_{语言对}_{日期}.txt
    """
    language_pair = get_language_pair(input_file_name)
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d%H%M")  # 格式：YYYYMMDDHHMM
    return f"{model}_{language_pair}_{date_str}.txt"

def print_results(source, candidates, best_index, rewards, best_candidate):
        # 输出结果
        print(f"源文本: {source}")
        print("候选翻译：")
        for i, candidate in enumerate(candidates):
            print(f"候选{i+1}: {candidate}")

        print("\n每个候选的reward：")
        for i, reward in enumerate(rewards):
            print(f"候选{i+1} reward分数: {reward:.4f}")

        print(f"\n最优候选: 候选{best_index + 1} - {best_candidate}")
        print()


def prm_selection(llm_instance, lang, source, hypotheses):
    rewards = []
    for i, hypothesis in enumerate(hypotheses):
        score = llm_instance.score(language_name=lang, src=source, mt=hypothesis, method='weighted')
        rewards.append(score)
    max_reward_index = rewards.index(max(rewards))
    return hypotheses[max_reward_index], max_reward_index, rewards

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="orm rewarding。")
    parser.add_argument("input_file", type=str, help="输入文件路径，文件名应包含语言对（如 zh-en 或 en-zh）。")
    parser.add_argument("--model", type=str, choices=["PRM-llama3B","PRM-qwen3B"], required=True, help="选择 PRM。")
    args = parser.parse_args()


    # 生成输出文件名
    output_file = "../result/" + get_output_file_name(args.input_file, args.model)

    # 读取输入
    input_file_path = "../data/source/" + args.input_file
    examples = read_input(input_file_path)

    llm_instance = MT()

    language_pair = get_language_pair(args.input_file)

    if language_pair == "zh-en":
        lang = llm_instance.language_name[4]
    elif language_pair == "en-zh":
        lang = llm_instance.language_name[5]

    # 处理每例
    for example in examples:
        #打印当前处理第几例
        print(examples.index(example) + 1)
        source = example["source"]
        hypotheses = example["hypotheses"]
        with open(output_file, "a", encoding="utf-8") as f:
            # 调用 DeepSeek API 选择最佳翻译
            try:
                best_hypothesis, best_index, rewards = prm_selection(llm_instance, lang, source, hypotheses)
                print_results(source, hypotheses, best_index, rewards, best_hypothesis)

                # 将结果写入文件
                f.write(f"{source}\n")
                f.write(f"{best_hypothesis}\n")
            except Exception as e:
                print(f"Error processing: {source}, Error: {str(e)}")
                # 如果出错，写入错误信息
                f.write(f"{source}\n")
                f.write(f"Error: {str(e)}\n")
                f.write("\n")  # 每例之间用空行分隔

    print(f"所有结果已保存到 {output_file}")

if __name__ == '__main__':
    main()

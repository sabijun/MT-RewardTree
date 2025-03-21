import argparse
import datetime
from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoModelForCausalLM


device = "cuda:5"
#model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"

#model_name = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

#model_id = "/mnt/zju105100171/home/jiahan/model/Llama3.2-3B-Instruct/rm/full/epoch2"
#model_id = "/mnt/zju105100171/home/jiahan/model/Llama3.2-3B-Instruct/rm/full_contrast/epoch2"       

class ArmoRMPipeline:
    def __init__(self, model_id, device_map="cuda:0", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}
    
def orm_selection(source, hypotheses, model_selection):
    rewards = []

    if model_selection == "ORM-Skywork-Reward-Llama-8B":
        model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

        for i, hypothesis in enumerate(hypotheses):
            conv = [{"role": "user", "content": source}, {"role": "assistant", "content": hypothesis}]
            conv_formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
            conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)
            with torch.no_grad():
                score = rm(**conv_tokenized).logits[0][0].item()
                rewards.append(score)

    elif model_selection == "ORM-ArmoRM-Llama3-8B-v0.1":
        prompt = source
        rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)

        for i, hypothesis in enumerate(hypotheses):
            response = hypothesis
            score = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            print(score)
            rewards.append(score["score"])

    elif model_selection == "ORM-internlm2-7b-reward":
        model = AutoModel.from_pretrained(
            "internlm/internlm2-7b-reward", 
            device_map="cuda:0", 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-7b-reward", trust_remote_code=True)

        for i, hypothesis in enumerate(hypotheses):
            chat = [
                {"role": "user", "content": source},
                {"role": "assistant", "content": hypothesis}
            ]
            score = model.get_score(tokenizer, chat)
            rewards.append(score)

    elif model_selection == "Ours-llama-3B-Contrast":
        model_name = "/home/jiahan/model/Llama3.2-3B-Instruct/rm/full_contrast/epoch2" 
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

        for i, hypothesis in enumerate(hypotheses):
            conv = [{"role": "user", "content": source}, {"role": "assistant", "content": hypothesis}]
            conv_formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
            conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)
            with torch.no_grad():
                score = rm(**conv_tokenized).logits[0][0].item()
                rewards.append(score)
    elif model_selection == "Ours-llama-3B-Token-DPO":
        model_name = "/mnt/zju105100171/home/jiahan/model/Llama3.2-3B-Instruct/rm/full/epoch2"
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

        for i, hypothesis in enumerate(hypotheses):
            conv = [{"role": "user", "content": source}, {"role": "assistant", "content": hypothesis}]
            conv_formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
            conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)
            with torch.no_grad():
                score = rm(**conv_tokenized).logits[0][0].item()
                rewards.append(score)
    elif model_selection == "Ours-Qwen-epoch1":
        model_name = "/home/jiahan/model/Qwen2.5-3B-Instruct/rm/full_contrast/epoch1"
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

        for i, hypothesis in enumerate(hypotheses):
            conv = [{"role": "user", "content": source}, {"role": "assistant", "content": hypothesis}]
            conv_formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
            conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)
            with torch.no_grad():
                score = rm(**conv_tokenized).logits[0][0].item()
                rewards.append(score)
    elif model_selection == "Ours-Qwen-epoch2":
        model_name = "/home/jiahan/model/Qwen2.5-3B-Instruct/rm/full/epoch2"
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

        for i, hypothesis in enumerate(hypotheses):
            conv = [{"role": "user", "content": source}, {"role": "assistant", "content": hypothesis}]
            conv_formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
            conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)
            with torch.no_grad():
                score = rm(**conv_tokenized).logits[0][0].item()
                rewards.append(score)

    max_reward_index = rewards.index(max(rewards))
    return hypotheses[max_reward_index], max_reward_index, rewards


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
            print(f"候选{i+1} Reward分数: {reward:.4f}")

        print(f"\n最优候选: 候选{best_index + 1} - {best_candidate}")
        print()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="orm rewarding。")
    parser.add_argument("input_file", type=str, help="输入文件路径，文件名应包含语言对（如 zh-en 或 en-zh）。")
    parser.add_argument("--model", type=str, choices=["ORM-Skywork-Reward-Llama-8B", "ORM-ArmoRM-Llama3-8B-v0.1", "ORM-internlm2-7b-reward", "Ours-llama-3B-Contrast", "Ours-llama-3B-Token-DPO", "Ours-Qwen-epoch1", "Ours-Qwen-epoch2"], required=True, help="选择 ORM。")
    args = parser.parse_args()

    language_pair = get_language_pair(args.input_file)
    if language_pair == "zh-en":
        language_name = ["Chinese", "English"]
    elif language_pair == "en-zh":
        language_name = ["English", "Chinese"]

    # 生成输出文件名
    output_file = "../result/" + get_output_file_name(args.input_file, args.model)

    # 读取输入
    input_file_path = "../data/source/" + args.input_file
    examples = read_input(input_file_path)

    # 处理每例
    for example in examples:
        #打印当前处理第几例
        print(examples.index(example) + 1)
        source = example["source"]
        hypotheses = example["hypotheses"]

        prompt = "Translate the following text from " + language_name[0] + " into " + language_name[1] + ".\n" + language_name[0] + ": " + source + "\n" + language_name[1] + ":"
        with open(output_file, "a", encoding="utf-8") as f:
            # 调用 DeepSeek API 选择最佳翻译
            try:
                best_hypothesis, best_index, rewards = orm_selection(prompt, hypotheses, args.model)
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

if __name__ == "__main__":
    main()
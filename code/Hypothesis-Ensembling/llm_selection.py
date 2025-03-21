import argparse
import datetime
import time
from openai import OpenAI
import google.generativeai as genai
import os
from google import genai

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'

# DeepSeek API 配置
DEEPSEEK_API_KEY = ""
DEEPSEEK_API_BASE_URL = ""

GEMINI_API_KEY = ""

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

def create_prompt(language_pair, source, hypotheses):
    """
    创建 prompt，用于与模型进行交互。
    """

    if language_pair == "zh-en":
        target_language = "English"
        source_language = "Chinese"
    elif language_pair == "en-zh":
        target_language = "Chinese"
        source_language = "English"
    prompt = f"This is a multiple choice question, choose a single answer. What is the best {target_language} translation for this {source_language} sentence?\n"
    prompt += f"Source: {source}\n"
    for i, hypothesis in enumerate(hypotheses, 1):
        prompt += f"Option {i}. {hypothesis}\n"
    prompt += "Correct answer: Option"

    return prompt

def call_gemini_api(prompt):
    """
    调用 Gemini API，选择最佳翻译。
    """
    print(f"Prompt: {prompt}")

    genai.configure(api_key='AIzaSyCYYx2O3L9HSpVOwWKyx-B_wBAzHXkdz3w')  # 填入自己的api_key
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    retries = 0
    max_retries = 100

    retry_delay=2
    while retries < max_retries:
        try:
            response = model.generate_content(prompt)
            # 处理响应
            if response.text:
                return response.text.strip()  # 返回正确的 content
            else:
                print(f"重试 {retries + 1}/{max_retries}: 返回内容为空")
        except Exception as e:
            print(f"重试 {retries + 1}/{max_retries}: API 调用失败 - {e}")
        
         # 重试逻辑
        retries += 1
        if retries < max_retries:
            time.sleep(retry_delay)  # 延迟一段时间后重试

    # 如果达到最大重试次数仍未成功，返回默认值或抛出异常
    raise Exception(f"API 调用失败: 已达到最大重试次数 {max_retries}")
    return response.text

def call_gemini_15_flash_api(prompt):
    """
    调用 Gemini API，选择最佳翻译。
    """
    print(f"Prompt: {prompt}")

    client = genai.Client(api_key="AIzaSyCYYx2O3L9HSpVOwWKyx-B_wBAzHXkdz3w")

    
    retries = 0
    max_retries = 100

    retry_delay=2
    while retries < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash", contents=prompt
            )
            # 处理响应
            if response.text:
                return response.text.strip()  # 返回正确的 content
            else:
                print(f"重试 {retries + 1}/{max_retries}: 返回内容为空")
        except Exception as e:
            print(f"重试 {retries + 1}/{max_retries}: API 调用失败 - {e}")
        
         # 重试逻辑
        retries += 1
        if retries < max_retries:
            time.sleep(retry_delay)  # 延迟一段时间后重试

    # 如果达到最大重试次数仍未成功，返回默认值或抛出异常
    raise Exception(f"API 调用失败: 已达到最大重试次数 {max_retries}")
    return response.text

def call_gemini_20_flash_api(prompt):
    """
    调用 Gemini API，选择最佳翻译。
    """
    print(f"Prompt: {prompt}")

    client = genai.Client(api_key="AIzaSyCYYx2O3L9HSpVOwWKyx-B_wBAzHXkdz3w")

    
    retries = 0
    max_retries = 100

    retry_delay=2
    while retries < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            # 处理响应
            if response.text:
                return response.text.strip()  # 返回正确的 content
            else:
                print(f"重试 {retries + 1}/{max_retries}: 返回内容为空")
        except Exception as e:
            print(f"重试 {retries + 1}/{max_retries}: API 调用失败 - {e}")
        
         # 重试逻辑
        retries += 1
        if retries < max_retries:
            time.sleep(retry_delay)  # 延迟一段时间后重试

    # 如果达到最大重试次数仍未成功，返回默认值或抛出异常
    raise Exception(f"API 调用失败: 已达到最大重试次数 {max_retries}")
    return response.text
    
def call_deepseek_api(prompt, model):
    """
    调用 DeepSeek API，选择最佳翻译。
    """

    print(f"Prompt: {prompt}")

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE_URL,
    )

    retries = 0
    max_retries = 100

    retry_delay=2
    while retries < max_retries:
        try:
            # 构造请求数据
            response = client.chat.completions.create(
                model=model,  # 指定模型
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                stream=False,  # 非流式输出
            )

            # 处理响应
            if response.choices:
                # 获取第一个 choice 的内容
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    if content:
                        return content.strip()
                    else:
                        print(f"重试 {retries + 1}/{max_retries}: 返回内容为空")
                else:
                    print(f"重试 {retries + 1}/{max_retries}: 返回结果结构无效")
            else:
                print(f"重试 {retries + 1}/{max_retries}: 返回结果为空")
        except Exception as e:
            print(f"重试 {retries + 1}/{max_retries}: API 调用失败 - {e}")
        
         # 重试逻辑
        retries += 1
        if retries < max_retries:
            time.sleep(retry_delay)  # 延迟一段时间后重试

    # 如果达到最大重试次数仍未成功，返回默认值或抛出异常
    raise Exception(f"API 调用失败: 已达到最大重试次数 {max_retries}")

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

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="从 DeepSeek API 选择最佳翻译并保存结果。")
    parser.add_argument("input_file", type=str, help="输入文件路径，文件名应包含语言对（如 zh-en 或 en-zh）。")
    parser.add_argument("--model", type=str, choices=["deepseek-r1", "deepseek-v3", "gemini", "gemini-2.0-flash", "gemini-1.5-flash"], required=True, help="选择模型（deepseek-r1, deepseek-v3, gemini, gemini-2.0-flash, gemini-1.5-flash）。")
    args = parser.parse_args()

    language_pair = get_language_pair(args.input_file)

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
        prompt = create_prompt(language_pair, source, hypotheses)
        with open(output_file, "a", encoding="utf-8") as f:
            # 调用 DeepSeek API 选择最佳翻译
            try:
                if args.model == "deepseek-r1" or args.model == "deepseek-v3":
                    print("Calling DeepSeek API...")
                    best_option = call_deepseek_api(prompt, args.model)
                elif args.model == "gemini":
                    print("Calling Gemini API...")
                    best_option = call_gemini_api(prompt)
                elif args.model == "gemini-2.0-flash":
                    print("Calling Gemini 2.0 Flash API...")
                    best_option = call_gemini_20_flash_api(prompt)
                elif args.model == "gemini-1.5-flash":
                    print("Calling Gemini 1.5 Flash API...")
                    best_option = call_gemini_15_flash_api(prompt)
                else:
                    raise ValueError(f"未知模型: {args.model}")
                if best_option is None:
                    raise Exception("API 返回结果为空")

                # 从返回文本中提取选项编号
                import re
                match = re.search(r"Option (\d+)", best_option)
                if not match:
                    raise Exception(f"API 返回的选项无效: {best_option}")

                best_index = int(match.group(1)) - 1
                if best_index < 0 or best_index >= len(hypotheses):
                    raise Exception(f"API 返回的选项无效: {best_option}")

                best_hypothesis = hypotheses[best_index]
                print(f"Processed: {source} -> {best_hypothesis}")

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
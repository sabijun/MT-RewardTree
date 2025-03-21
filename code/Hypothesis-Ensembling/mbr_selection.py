import sys
import datetime
import torch
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from comet import download_model, load_from_checkpoint

# 全局变量：COMET 模型
COMET_MODEL = None

def load_comet_model():
    """
    加载 COMET 模型（全局只加载一次）
    """
    global COMET_MODEL
    if COMET_MODEL is None:
        model_path = download_model("Unbabel/wmt22-comet-da")  # 下载 COMET 模型
        COMET_MODEL = load_from_checkpoint(model_path)  # 加载模型
    return COMET_MODEL

def calculate_comet_loss(source, candidate, reference, lang="zh"):
    """
    计算COMET损失函数
    """
    # 加载 COMET 模型（全局只加载一次）
    comet_model = load_comet_model()

    data = [
        {
            "src": source,
            "mt": candidate,
            "ref": reference
        }
    ]
    output = comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    score = output.scores[0]
    
    return 1 - score

    

def calculate_bleu_loss(source, candidate, reference, lang="zh"):
    """
    计算BLEU损失函数
    """
    return 1 - sentence_bleu([reference], candidate)

def calculate_bertscore_loss(source, candidate, reference, lang="zh"):
    """
    计算BERTScore损失函数，支持中文和英文
    """
    # 根据语言选择模型
    if lang == "zh":
        model_type = "bert-base-chinese"  # 中文模型
    elif lang == "en":
        model_type = "bert-base-uncased"  # 英文模型
    else:
        raise ValueError("Unsupported language. Choose 'zh' for Chinese or 'en' for English.")

    # 设置设备（GPU 或 CPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 计算BERTScore，并将模型和数据加载到指定设备
    P, R, F1 = bert_score(
        [candidate], 
        [reference], 
        lang=lang, 
        model_type=model_type,
        device=device  # 指定设备
    )
    
    # 返回1 - F1作为损失
    return 1 - F1.mean().item()

def calculate_pairwise_loss(source, candidates, loss_function, lang):
    """
    计算候选翻译之间的成对损失
    """
    N = len(candidates)
    loss_matrix = [[0.0] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                loss_matrix[i][j] = 0
            else:
                loss_matrix[i][j] = loss_function(source, candidates[i], candidates[j], lang)
    #print(loss_matrix)
    return loss_matrix

def calculate_average_risk(loss_matrix):
    """
    计算每个候选翻译的平均风险
    """
    N = len(loss_matrix)
    average_risks = []
    
    for i in range(N):
        total_loss = sum(loss_matrix[i])  # 计算第i个候选的总损失
        average_risk = total_loss / (N - 1)  # 平均风险（排除自身）
        average_risks.append(average_risk)
    
    return average_risks

def select_best_candidate(candidates, average_risks):
    """
    选择平均风险最小的候选翻译
    """
    min_risk_index = average_risks.index(min(average_risks))
    return candidates[min_risk_index], min_risk_index

def mbr_selection(source, candidates, loss_function, lang):
    loss_matrix = calculate_pairwise_loss(source, candidates, loss_function, lang)
    average_risks = calculate_average_risk(loss_matrix)
    best_candidate, best_index = select_best_candidate(candidates, average_risks)

    return best_candidate, best_index, average_risks

def get_inputs():

    if len(sys.argv) != 3:
        print("Usage: python mbr_selection.py <input_file> <loss_function>")
        sys.exit(1)
    
    file_name = sys.argv[1]
    input_file = "../data/source/" + file_name
    loss_function_name = sys.argv[2]
    
    if loss_function_name == "bleu":
        loss_function = calculate_bleu_loss
    elif loss_function_name == "bert":
        loss_function = calculate_bertscore_loss
    elif loss_function_name == "comet":
        loss_function = calculate_comet_loss
    else:
        print("Invalid loss function. Choose 'bleu' or 'bert'.")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    return file_name, lines, loss_function, loss_function_name

def print_results(source, candidates, best_index, average_risks, best_candidate):
        # 输出结果
        print(f"源文本: {source}")
        print("候选翻译：")
        for i, candidate in enumerate(candidates):
            print(f"候选{i+1}: {candidate}")

        print("\n每个候选的平均风险：")
        for i, risk in enumerate(average_risks):
            print(f"候选{i+1} 平均风险: {risk:.4f}")

        print(f"\n最优候选: 候选{best_index + 1} - {best_candidate}")
        print()

def write_to_file(output_file, source, best_candidate):
    
    with open(output_file, 'a', encoding='utf-8') as f:  # 使用 'a' 模式追加写入
        f.write(f"{source}\n{best_candidate}\n")

def main():
    
    input_file_name, lines, loss_function, loss_function_name = get_inputs()

    lang = input_file_name.split("-")[1]

    # 获取当前日期和时间
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d%H%M")
    output_file = f'../result/mbr_{loss_function_name}_{input_file_name}_{date_str}.txt'
    
    # 清空文件（如果文件已存在）
    with open(output_file, 'w', encoding='utf-8') as f:
        pass  # 清空文件内容

    
    for i in range(0, len(lines), 9):
        print(f"第{i // 9 + 1}组数据:")
        source = lines[i].strip()
        candidates = [line.strip() for line in lines[i + 1:i + 9]]

        best_candidate, best_index, average_risks = mbr_selection(source, candidates, loss_function, lang)
        
        # 将最佳翻译写入文件
        write_to_file(output_file, source, best_candidate)

        # 打印结果
        print_results(source, candidates, best_index, average_risks, best_candidate)
    
    print(f"所有结果已保存")

if __name__ == "__main__":
    main()
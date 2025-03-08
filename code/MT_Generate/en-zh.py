import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from comet import load_from_checkpoint
from typing import List
import logging


class MT:
    def __init__(self, llm_model, llm_path, torch_dtype=torch.float16):
        self.LLM = llm_model.eval()
        self.eva = eva
        self.llm_tokenizer=AutoTokenizer.from_pretrained(llm_path)
        self.topk = 2
        self.max_new_token = 128
        self.temperature = 1
        self.debug = False
        self.threshold = [0.04, 0.4]
    
    def tokens_to_text(self, tokenizer, tokens: torch.Tensor, skip_special_tokens=True) -> List[str]:
        return tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def get_input_ids(self,tokenizer, prompt:str, template:str = 'it',add_generation_prompt: bool = False) -> torch.Tensor:
        if template == 'it':
            tokens = tokenizer(prompt, return_tensors="pt").input_ids
        elif template == 'chat':
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
    
    def get_score(self, src: str, results: list):
        score_data = []
        for sentence in results:
            if self.debug:
                print(sentence)
            temp_dic = {"src": src, "mt": sentence[0]}
            score_data.append(temp_dic)
        eva_output = eva.predict(score_data, batch_size = 6,gpus=1,progress_bar=False)
        return eva_output['scores']

    
    def process(self, src: str):
        prompt = [{"role": "user", "content": "Translate the following text from English into Chinese.\nEnglish: " + src + "\nChinese:"}]
        prompt_tokens = self.get_input_ids(tokenizer=self.llm_tokenizer,prompt=prompt,template='chat',add_generation_prompt=True).to(llm_device)
        end_label = False
        prompt_length = len(prompt_tokens[0])
        
        gap = []
        high_mean = []
        perference_pair = []
        score_pair = []
        mean_score_list = []
        full_score = []
        
        while not end_label:
            same_flag = False
            candidate_tokens = []
            candidate_sentences = []
            results = []
            if self.debug:
                print(len(prompt_tokens[0]))
            #generate one new token
            attention_mask = torch.ones(prompt_tokens.shape,dtype=torch.int8).to(llm_device)
            gen_tokens = self.LLM.generate(     
                prompt_tokens,
                temperature=0.95,
                top_p=0.95,
                top_k=50,
                max_new_tokens=1,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            logits, tokens = torch.topk(gen_tokens.scores[0][0],dim=0,k=2)
            # resize the tensor, and add to the candidate list
            if logits[1] == -torch.inf:
                candidate_tokens=[tokens[0], tokens[0]]
                same_flag = True
            else:
                candidate_tokens=[tokens[0], tokens[1]]
            candidate_sentences.append(torch.cat((prompt_tokens,candidate_tokens[0].unsqueeze(0).unsqueeze(0)),1))
            candidate_sentences.append(torch.cat((prompt_tokens,candidate_tokens[1].unsqueeze(0).unsqueeze(0)),1))
            if self.debug:
                print(torch.sort(gen_tokens.scores[0][0],descending=True))
                print(f'top1\ntoken ID:{tokens[0]}\ttoken:{self.tokens_to_text(self.llm_tokenizer, [tokens[0]])}\tlogits:{logits[0]}')
                print(f'top2\ntoken ID:{tokens[1]}\ttoken:{self.tokens_to_text(self.llm_tokenizer, [tokens[1]])}\tlogits:{logits[1]}')

            for candidate in candidate_sentences:
                if candidate[0][-1] != self.llm_tokenizer.eos_token_id:
                    # for each candidate, sampling 3 sentences
                    attention_mask = torch.ones(candidate.shape,dtype=torch.int8).to(llm_device)
                    gen_tokens = self.LLM.generate(       
                        candidate,
                        temperature=0.95,
                        top_p=0.95,
                        top_k=50,
                        max_new_tokens=512,
                        do_sample=True,
                        output_scores=True,
                        return_dict_in_generate=True,
                        num_return_sequences=3,
                        attention_mask=attention_mask,
                        pad_token_id=self.llm_tokenizer.eos_token_id
                    )
                    for i in range(3):
                        results.append(self.tokens_to_text(self.llm_tokenizer, gen_tokens.sequences[i][prompt_length:].unsqueeze(0)))
                else:
                    for _ in range(3):
                        results.append(self.tokens_to_text(self.llm_tokenizer, candidate[0][prompt_length:].unsqueeze(0)))
            
            score = self.get_score(src, results)
            score1 = sum(score[0:3]) / 3
            score2 = sum(score[3:6]) / 3
            full_score.append(score)
            if same_flag:
                gap.append(0)
            else:
                gap.append(abs(score1 - score2))
            mean_score_list.append([score1, score2])
            high_mean.append(score1 if score1 > score2 else score2)
            temp = []
            temp_score = []
            if score1 > score2:
                temp = [results[torch.argsort(torch.tensor(score[0:3]), descending=True)[0]], results[torch.argsort(torch.tensor(score[3:6]))[1] + 3]]
                temp_score = [score[torch.argsort(torch.tensor(score[0:3]), descending=True)[0]], score[torch.argsort(torch.tensor(score[3:6]))[1] + 3]]
            else:
                temp = [results[torch.argsort(torch.tensor(score[3:6]), descending=True)[0] + 3], results[torch.argsort(torch.tensor(score[0:3]))[1]]]
                temp_score = [score[torch.argsort(torch.tensor(score[3:6]), descending=True)[0] + 3], score[torch.argsort(torch.tensor(score[0:3]))[1]]]
            perference_pair.append(temp)
            score_pair.append(temp_score)
            
            perference_token = candidate_tokens[0] if score1 > score2 else candidate_tokens[1]
            # if is end
            if perference_token == self.llm_tokenizer.eos_token_id:
                end_label = True
            prompt_tokens = torch.cat((prompt_tokens,perference_token.unsqueeze(0).unsqueeze(0)), 1)

        
        rank = torch.argsort(torch.tensor(gap),descending=True)
        print(gap[rank[0]])
        print(perference_pair[rank[0]][0][0]+'\t'+str(score_pair[rank[0]][0]))
        print(perference_pair[rank[0]][1][0]+'\t'+str(score_pair[rank[0]][1]))

        if gap[rank[0]] >= self.threshold[0] and gap[rank[0]] <= self.threshold[1]:
            with open(OUTPUT_PATH + 'prompt4_max_gap','a') as f:
                f.write(src+'\n')
                f.write(perference_pair[rank[0]][0][0]+'\t'+str(score_pair[rank[0]][0])+'\n')
                f.write(perference_pair[rank[0]][1][0]+'\t'+str(score_pair[rank[0]][1])+'\n')
                f.close()
        with open(OUTPUT_PATH + 'prompt4_debug','a') as f:
            f.write(src+'\n')
            f.write(str(gap[rank[0]])+'\n')
            f.write(str(perference_pair[rank[0]][0][0])+'\n')
            f.write(str(perference_pair[rank[0]][1][0])+'\n')
            f.write(str(gap)+'\n')
            f.write(str(mean_score_list)+'\n')
            f.write(str(high_mean)+'\n')
            f.write(str(perference_pair)+'\n')
            f.write(str(score_pair)+'\n')
            f.write(str(full_score)+'\n')
            f.close()

        
        rank = torch.argsort(torch.tensor(high_mean), descending=True)
        with open(OUTPUT_PATH + 'prompt4_max_mean','a') as f:
            f.write(src+'\n')
            f.write(perference_pair[rank[0]][0][0]+'\t'+str(score_pair[rank[0]][0])+'\n')
            f.write(perference_pair[rank[0]][1][0]+'\t'+str(score_pair[rank[0]][1])+'\n')
            f.close()
            
        if self.debug:
            print(perference_pair)

if __name__ == '__main__':
    # 消除warning
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    #加载模型
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    LLM_PATH = "/home/jiahan/repository/TowerInstruct-7B-v0.2"
    EVA_PATH = "/home/jiahan/repository/wmt22-cometkiwi-da/checkpoints/model.ckpt"
    OUTPUT_PATH = "/home/jiahan/results/preference_pair_benchwork/tower_it_v0.2/en-zh/max_median/"
    DATA_PATH = "/home/jiahan/repository/alma/human_written_data/zhen/train.zh-en.en"

    llm_device= "cuda:0"

    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, torch_dtype=torch.bfloat16).to(llm_device)
    eva = load_from_checkpoint(EVA_PATH).to(llm_device)

    source_num = 5000
    cnt = 0
    start_idx = 1955
    llm_instance = MT(llm,LLM_PATH)
    llm_instance.debug = False


    source_data = []
    with open(DATA_PATH,'r') as f:
        for _ in range(source_num):
            source_data.append(f.readline().strip())
        f.close()

    i = start_idx
    while True:
        print(f'{i}/{source_num}:')
        src = source_data[i]
        print(src)
        llm_instance.process(src)
        i += 1
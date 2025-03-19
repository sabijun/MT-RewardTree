import argparse
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from comet import download_model, load_from_checkpoint
from typing import List


# 消除warning
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MT:
    def __init__(self, llm, tokenizer, eva, threshold, output_path, lang_pair, torch_dtype=torch.float16):
        self.llm = llm.eval()
        self.eva = eva
        self.tokenizer=tokenizer
        self.temperature = 1
        self.topk = 2
        self.max_new_token = 512   
        self.threshold = threshold
        self.lang_pair = lang_pair
        self.output_path = output_path
        self.device = next(llm.parameters()).device
        self.debug = False   

        dir_path = os.path.dirname(output_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

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
        eva_output = self.eva.predict(score_data, batch_size = 6,gpus=1,progress_bar=False)
        return eva_output['scores']

    
    def process(self, src: str):
        prompt = [{"role": "user", "content": "Translate the following text from " + self.lang_pair[0] + " into " + self.lang_pair[1] + ".\n" + self.lang_pair[0] + ":" + src + "\n" + self.lang_pair[1] + ":"}]
        prompt_tokens = self.get_input_ids(tokenizer=self.tokenizer,prompt=prompt,template='chat',add_generation_prompt=True).to(self.device)
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
            attention_mask = torch.ones(prompt_tokens.shape,dtype=torch.int8).to(self.device)
            gen_tokens = self.llm.generate(     
                prompt_tokens,
                temperature=0.95,
                top_p=0.95,
                top_k=50,
                max_new_tokens=1,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id
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
                print(f'top1\ntoken ID:{tokens[0]}\ttoken:{self.tokens_to_text(self.tokenizer, [tokens[0]])}\tlogits:{logits[0]}')
                print(f'top2\ntoken ID:{tokens[1]}\ttoken:{self.tokens_to_text(self.tokenizer, [tokens[1]])}\tlogits:{logits[1]}')

            for candidate in candidate_sentences:
                if candidate[0][-1] != self.tokenizer.eos_token_id:
                    # for each candidate, sampling 3 sentences
                    attention_mask = torch.ones(candidate.shape,dtype=torch.int8).to(self.device)
                    gen_tokens = self.llm.generate(       
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
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    for i in range(3):
                        results.append(self.tokens_to_text(self.tokenizer, gen_tokens.sequences[i][prompt_length:].unsqueeze(0)))
                else:
                    for _ in range(3):
                        results.append(self.tokens_to_text(self.tokenizer, candidate[0][prompt_length:].unsqueeze(0)))
            
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
            if perference_token == self.tokenizer.eos_token_id:
                end_label = True
            prompt_tokens = torch.cat((prompt_tokens,perference_token.unsqueeze(0).unsqueeze(0)), 1)

        
        rank = torch.argsort(torch.tensor(gap),descending=True)
        print(src)
        print(gap[rank[0]])
        print(perference_pair[rank[0]][0][0]+'\t'+str(score_pair[rank[0]][0]))
        print(perference_pair[rank[0]][1][0]+'\t'+str(score_pair[rank[0]][1]))

        if gap[rank[0]] >= self.threshold[0] and gap[rank[0]] <= self.threshold[1]:
            with open(self.output_path,'a') as f:
                f.write(src+'\n')
                f.write(perference_pair[rank[0]][0][0]+'\t'+str(score_pair[rank[0]][0])+'\n')
                f.write(perference_pair[rank[0]][1][0]+'\t'+str(score_pair[rank[0]][1])+'\n')
                f.close()
                    
        if self.debug:
            print(perference_pair)

if __name__ == '__main__':
    lang_name = {'en':'English', 'de':'German', 'ru':'Russian', 'zh':'Chinese'}
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='the path of the src sentence file')
    parser.add_argument('output_file', type=str, help='the path of the output file')
    parser.add_argument('src_lang', type=str, help='the language of the src sentence')
    parser.add_argument('tar_lang', type=str, help='the translate target language')
    parser.add_argument('--gpu_map', type=str, default= '0,1')
    parser.add_argument('--trans_model', type=str, default='Unbabel/TowerInstruct-7B-v0.2', help='the path or name of the model')
    parser.add_argument('--eva_model', type=str, default='Unbabel/wmt22-cometkiwi-da', help='the name of the reference-free evaluation model')
    parser.add_argument('--threshold', type=int, nargs='+', default=[0.04, 0.4])

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_map
    llm = AutoModelForCausalLM.from_pretrained(args.trans_model, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.trans_model)
    threshold = args.threshold
    lang_pair = [lang_name[args.src_lang], lang_name[args.tar_lang]]
    output_path = args.output_file

    if args.eva_model[len(args.eva_model) - 5:] == '.ckpt':
        eva = load_from_checkpoint(args.eva_model)
    else:    
        model_path = download_model(args.eva_model)
        eva = load_from_checkpoint(model_path)
 
    mt = MT(llm, tokenizer, eva, threshold, output_path, lang_pair)
    
    f = open(args.input_file, 'r', encoding='utf-8')
    lines = f.read().splitlines()
    f.close()

    for line in lines:
        mt.process(line)


    
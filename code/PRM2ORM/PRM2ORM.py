import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


class PRM2ORM:
    def __init__(self, trained_model, trained_device, ref_model, ref_device, tokenizer):
        self.trained_LLM = trained_model.eval()
        self.ref_LLM = ref_model.eval()
        self.trained_device = trained_device
        self.ref_device = ref_device
        self.llm_tokenizer= tokenizer
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
        mt_token_list = self.get_input_ids(tokenizer=self.llm_tokenizer,prompt=mt_prompt,template='chat',add_generation_prompt=True).to(self.trained_device)
        prompt_token_list = self.get_input_ids(tokenizer=self.llm_tokenizer,prompt=prompt,template='chat',add_generation_prompt=True).to(self.trained_device)
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

            prompt_token_list = prompt_token_list.to(self.trained_device)
            attention_mask = torch.ones(prompt_token_list.shape,dtype=torch.int8).to(self.trained_device)
            trained_results = self.trained_LLM.generate(     
                prompt_token_list,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                pad_token_id=self.llm_tokenizer.eos_token_id
                )   
            trained_results = torch.softmax(trained_results.scores[0],dim = 1)
            trained_probability = trained_results[0][mt_token_list[0][prompt_length + i]]

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
            ref_probability = ref_results[0][mt_token_list[0][prompt_length + i]].to(self.trained_device)
            
            if method == 'weighted':
                reward += torch.log(trained_probability / ref_probability + 0.1) / (i + 1)
            elif method == 'length normalized':
                reward += torch.log(trained_probability / ref_probability + 0.1) / length
            else:
                raise ValueError("Unknown Method")
            prompt_token_list = prompt_token_list.to(self.trained_device)
            prompt_token_list = torch.cat((prompt_token_list,mt_token_list[0][prompt_length + i].unsqueeze(0).unsqueeze(0)),dim=1)
        return reward
    

if __name__ == '__main__':
    lang_name = {'en':'English', 'de':'German', 'ru':'Russian', 'zh':'Chinese'}
    parser = argparse.ArgumentParser()

    parser.add_argument('trained_model', type=str)
    parser.add_argument('ref_model', type=str)
    parser.add_argument('input_path', type=str)
    parser.add_argument('src_lang', type=str)
    parser.add_argument('tar_lang', type=str)

    parser.add_argument('--trained_device', type=str, default='0')
    parser.add_argument('--ref_device', type=str, default='1')
    parser.add_argument('--method', type=str, default='weighted')

    args = parser.parse_args()

    trained_model = AutoModelForCausalLM.from_pretrained(args.trained_model, torch_dtype=torch.bfloat16).to('cuda:' + args.trained_device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model, torch_dtype=torch.bfloat16).to('cuda:' + args.ref_device)
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model)
    src_lang = args.src_lang
    tar_lang = args.tar_lang
    method = args.method

    f = open(args.input_path, 'r', encoding='utf-8')
    lines = f.read().splitlines()
    f.close()

    prm2orm = PRM2ORM(trained_model, 'cuda:' + args.trained_device, ref_model, 'cuda:' + args.ref_device, tokenizer)

    f = open('output/results', 'w', encoding='utf-8')
    for i in range(len(lines) // 2):
        score = prm2orm.score([lang_name[src_lang], lang_name[tar_lang]], lines[2*i], lines[2*i+1], method)
        f.write(lines[2*i] + '\n' + lines[2*i+1] + '\t' + str(score.item()) + '\n')
        print(lines[2*i] + '\n' + lines[2*i+1] + '\nscore:' + str(score.item()))
    f.close()
        

    

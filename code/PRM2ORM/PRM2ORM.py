import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class PRM2ORM:
    def __init__(self, dpo_path, ref_path, dpo_device, ref_device, torch_dtype=torch.float16):
        DPO_PATH = dpo_path
        REF_PATH = ref_path
        self.dpo_device = dpo_device
        self.ref_device = ref_device 
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
import torch
from torch import nn, optim
from peft import PeftModel, LoraConfig, get_peft_model

from transformers import (
    AutoModelForCausalLM, Blip2Model, BlipImageProcessor, AutoTokenizer, BitsAndBytesConfig
)

class BLIP2ForPhi(nn.Module):
    def __init__(self, vision_model, q_former, language_model, query_tokens):
        super().__init__()
        self.vision_model = vision_model
        self.q_former = q_former
        self.projection = nn.Linear(q_former.config.hidden_size, language_model.config.hidden_size)
        self.phi_model = language_model
        self.query_tokens = query_tokens


    
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        image_embeds = self.vision_model(pixel_values).last_hidden_state

        batch_size = image_embeds.shape[0]
        qformer_query_embeds = self.query_tokens.expand(batch_size, -1, -1)

        
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_outputs = self.q_former(
            query_embeds=qformer_query_embeds,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask
        )[0]

        projected_query = self.projection(query_outputs)

        text_embeds = self.phi_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([projected_query, text_embeds], dim=1)
        
        query_attention_mask = torch.ones(projected_query.size()[:-1], dtype=torch.long, device=projected_query.device)
        combined_attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)

        
        outputs = self.phi_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels, 
        )
        return outputs
    

    def generate(self, pixel_values, input_ids, attention_mask, **generate_kwargs):
        image_embeds = self.vision_model(pixel_values).last_hidden_state
        batch_size = image_embeds.shape[0]
        qformer_query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_outputs = self.q_former(
            query_embeds=qformer_query_embeds,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask
        )[0]

        projected_query = self.projection(query_outputs) # shape: [B, 32, D_phi]
        text_embeds = self.phi_model.get_input_embeddings()(input_ids) # shape: [B, S, D_phi]
        inputs_embeds = torch.cat([projected_query, text_embeds], dim=1)
        query_attention_mask = torch.ones(projected_query.size()[:-1], dtype=torch.long, device=projected_query.device)
        combined_attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)

        outputs = self.phi_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            **generate_kwargs
        )
        return outputs
    


def setup_model(config):
    vision_model_name = config['model']['vision_model_name']
    llm_name = config['model']['llm_name']

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    image_processor = BlipImageProcessor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    blip2_model = Blip2Model.from_pretrained(vision_model_name)
    vision_model = blip2_model.vision_model
    q_former = blip2_model.qformer
    query_tokens = blip2_model.query_tokens

    phi_model = AutoModelForCausalLM.from_pretrained(
        llm_name, quantization_config=quantization_config, trust_remote_code=True
    )

    model = BLIP2ForPhi(vision_model, q_former, phi_model, query_tokens)

    return model, image_processor, tokenizer


def select_train_params(model, qformer=True, projection=True, language_model=True):
    for param in model.vision_model.parameters():
        param.requires_grad = False

    if qformer:
        for param in model.q_former.parameters():
            param.requires_grad = True
    else:
        for param in model.q_former.parameters():
            param.requires_grad = False
    if projection:
        for param in model.projection.parameters():
            param.requires_grad = True
    else:
        for param in model.projection.parameters():
            param.requires_grad = False

    if not language_model:
        for param in model.phi_model.parameters():
            param.requires_grad = False

    trainable_param_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return len(trainable_param_names)

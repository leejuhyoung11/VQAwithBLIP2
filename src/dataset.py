from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import Dataset as HFDataset
from collections import Counter
from tqdm import tqdm
import random, os, re
import pandas as pd

class ImageCaptioningDataset(Dataset):
    def __init__(self, hf_dataset, image_processor, tokenizer, num_query_tokens=32, max_length=128, is_train=True):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_query_tokens = num_query_tokens
        self.is_train = is_train

        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # 정규화
            ])

        else: 
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item['image'].convert("RGB")
            
            #1 cocoCaption
            if 'answer' in item and item['answer']:
                captions = item['answer']
                caption = captions[0].replace('\n', ' ').strip()
            #2 flickr30k
            elif 'caption' in item and item['caption']:
                captions_list = item.get('caption')
                if captions_list:
                    valid_captions = [c.strip().replace('\n', ' ') for c in captions_list if c and c.strip()]
                if valid_captions:
                    caption = valid_captions[0]
            #3 textcaps
            elif 'caption_str' in item and item['caption_str']:
                captions_list = item.get('caption_str')
                if captions_list:
                    valid_captions = [c.strip().replace('\n', ' ') for c in captions_list if c and c.strip()]
                if valid_captions:
                    caption = valid_captions[0]
            #LLaVa-Recap
            elif 'conversations' in item and item['conversations']:
                for turn in item['conversations']:
                    if turn.get('from') == 'gpt':
                        caption = turn.get('value')
                        break
            
            if caption is None or image is None:
                print(f"No Element in {idx}")
                return None
            
            
            pixel_values = self.transforms(image)
            prompt = "Describe the image.\nAnswer :"
            
            len_of_prompt = len(self.tokenizer(prompt)['input_ids'])

            inputs = self.tokenizer(
                prompt + caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            text_labels = inputs.input_ids.clone()
            text_labels[:, :len_of_prompt] = -100
            text_labels[text_labels == self.tokenizer.pad_token_id] = -100
            
            query_labels = torch.full((1, self.num_query_tokens), -100)

            combined_labels = torch.cat([query_labels, text_labels], dim=1)

            return {
                "pixel_values": pixel_values.squeeze(),
                "input_ids": inputs.input_ids.squeeze(),   
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": combined_labels.squeeze()
            }
        except Exception as e:
            print(f"Whil processing index {idx} , error occured ({e}), Skip Element.")
            return None


class VQADataset(Dataset):
    def __init__(self, hf_dataset, image_processor, tokenizer, num_query_tokens=32, max_length=128, is_train=True, image_dataset=None):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_query_tokens = num_query_tokens
        self.is_train = is_train
        self.image_dataset = image_dataset

        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # 정규화
            ])

        else: 
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            sample_type = "short"
            #GQA Dataset : short
            if self.image_dataset:
                image_id = item['imageId']
                image = self.image_dataset[image_id].convert("RGB")
                question = item['question']
                answer = item['fullAnswer']
                

            else:
                image = item['image'].convert("RGB")
                
                #VQAV2 Dataset : short
                if 'multiple_choice_answer' in item and item['multiple_choice_answer']:
                    question = item['question']
                    answer = item['multiple_choice_answer']
                # LLava Next-Data : long
                elif 'conversations' in item and item['conversations']:
                    sample_type = "long"
                    res = export_qna_from_conversation(item, seed=idx)
                    if res is None:
                        return None
                    question, answer = res
                # OKVQA Dataset : Reasoning
                else:
                    question = item['question']
                    if not item['answers']:
                        return None
                    sample_type = "reasoning"
                    answers = item['answers']
                    answer_counts = Counter(answers)
                    # Choose most common answer from the list
                    most_common_answer = answer_counts.most_common(1)
                    if most_common_answer:
                        answer = most_common_answer[0][0]
                    else:
                        return None
            
            # Exclude non-english data
            if contains_chinese(question) or contains_chinese(answer):
                return None

            pixel_values = self.transforms(image)

            prompt = (f"Question: {question}\n" +"Answer:")
            len_of_prompt = len(self.tokenizer(prompt)['input_ids'])

            inputs = self.tokenizer(
                prompt + answer,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            answer_tokens = inputs.input_ids.clone()
            answer_tokens[:, :len_of_prompt] = -100
            answer_tokens[answer_tokens == self.tokenizer.pad_token_id] = -100

            query_labels = torch.full((1, self.num_query_tokens), -100)
            combined_labels = torch.cat([query_labels, answer_tokens], dim=1)


            return {
                "pixel_values": pixel_values.squeeze(),
                "input_ids": inputs.input_ids.squeeze(),   
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": combined_labels.squeeze(),
                "sample_type": sample_type
            }
        except Exception as e:
            print(f"Whil processing index {idx} , error occured ({e}), Skip Element.")
            return None


class LlavaInstructDataset(Dataset):
    def __init__(self, hf_dataset, image_processor, tokenizer, num_query_tokens=32, max_length=128, is_train=True, image_dataset=None):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_query_tokens = num_query_tokens
        self.is_train = is_train
        self.image_dir = image_dataset
        self.image_name_prefix = "COCO_train2014_"

        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # 정규화
            ])

        else: 
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            sample_type = "reasoning"
            item = self.dataset[idx]
            image_name = self.image_name_prefix+item['image']
            image_path = os.path.join(self.image_dir, image_name)
            image = Image.open(image_path).convert("RGB")

            question, answer = export_qna_from_conversation(item, seed=idx)

            pixel_values = self.transforms(image)

            prompt = (f"Question: {question}\n" +"Answer:")
            len_of_prompt = len(self.tokenizer(prompt)['input_ids'])

            inputs = self.tokenizer(
                prompt + answer,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            answer_tokens = inputs.input_ids.clone()
            answer_tokens[:, :len_of_prompt] = -100
            answer_tokens[answer_tokens == self.tokenizer.pad_token_id] = -100

            query_labels = torch.full((1, self.num_query_tokens), -100)
            combined_labels = torch.cat([query_labels, answer_tokens], dim=1)


            return {
                "pixel_values": pixel_values.squeeze(),
                "input_ids": inputs.input_ids.squeeze(),   
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": combined_labels.squeeze(),
                "sample_type": sample_type
            }
        except Exception as e:
            print(f"Whil processing index {idx} , error occured ({e}), Skip Element.")
            return None


class AOKVQADataset(Dataset):
    def __init__(self, hf_dataset, image_processor, tokenizer, num_query_tokens=32, max_length=128, is_train=True, image_dataset=None):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_query_tokens = num_query_tokens
        self.is_train = is_train
        self.image_dataset = image_dataset

        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # 정규화
            ])

        else: 
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item['image'].convert("RGB")
            
            question = item['question']
            choices = item['choices']
            answer_idx = item['correct_choice_idx']
            
            letter = chr(65 + answer_idx)  
            answer = f"{letter}. {choices[answer_idx]}"

            pixel_values = self.transforms(image)

            prompt = (
                "You are a helpful AI that answers multiple-choice questions based on the given image.\n"
                "Select only the single best answer from A, B, C, or D.\n\n"
                "Respond only with one letter (e.g., 'A').\n\n"
                f"Question: {question}\n\n" +
                "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]) +
                "\n\nAnswer:"
            )

            len_of_prompt = len(self.tokenizer(prompt)['input_ids'])

            inputs = self.tokenizer(
                prompt + answer,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            answer_tokens = inputs.input_ids.clone()
            answer_tokens[:, :len_of_prompt] = -100
            answer_tokens[answer_tokens == self.tokenizer.pad_token_id] = -100

            query_labels = torch.full((1, self.num_query_tokens), -100)
            combined_labels = torch.cat([query_labels, answer_tokens], dim=1)


            return {
                "pixel_values": pixel_values.squeeze(),
                "input_ids": inputs.input_ids.squeeze(),   
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": combined_labels.squeeze(),
            }
        except Exception as e:
            print(f"Whil processing index {idx} , error occured ({e}), Skip Element.")
            return None


class Visual7wDataset(Dataset):
    def __init__(self, hf_dataset, image_processor, tokenizer, num_query_tokens=32, max_length=128, is_train=True, image_dataset=None):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_query_tokens = num_query_tokens
        self.is_train = is_train
        

        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # 정규화
            ])

        else: 
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            
            item = self.dataset[idx]
            image = item['image']
            prompt = item['question']
            answer = item['answer']

            pixel_values = self.transforms(image)
            len_of_prompt = len(self.tokenizer(prompt)['input_ids'])

            inputs = self.tokenizer(
                prompt + answer,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            answer_tokens = inputs.input_ids.clone()
            answer_tokens[:, :len_of_prompt] = -100
            answer_tokens[answer_tokens == self.tokenizer.pad_token_id] = -100

            query_labels = torch.full((1, self.num_query_tokens), -100)
            combined_labels = torch.cat([query_labels, answer_tokens], dim=1)


            return {
                "pixel_values": pixel_values.squeeze(),
                "input_ids": inputs.input_ids.squeeze(),   
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": combined_labels.squeeze(),
            }
        except Exception as e:
            print(f"Whil processing index {idx} , error occured ({e}), Skip Element.")
            return None



def get_captioning_datasets(dataset_name, image_processor, tokenizer, tokenizer_max_length=128):
    raw_datasets = load_dataset(dataset_name)
    # Use only first dict
    raw_dataset = list(raw_datasets.values())[0]

    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_raw_dataset = split_dataset['train']
    eval_raw_dataset = split_dataset['test']

    train_dataset = ImageCaptioningDataset(train_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length)
    valid_dataset = ImageCaptioningDataset(eval_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length)
    train_debug = Subset(train_dataset, indices=range(10))
    valid_debug = Subset(valid_dataset, indices=range(10))

    return train_dataset, valid_dataset, train_debug, valid_debug


def get_vqa_datasets(dataset_name, image_processor, tokenizer, tokenizer_max_length=128, img_dir=None):
    
    id_to_image = None
    # images and quesitons are seperated
    if isinstance(dataset_name, list):
        image_dataset = load_dataset(dataset_name[0], dataset_name[1][0])
        image_dataset = list(image_dataset.values())[0]
        id_to_image = {item['id']: item['image'] for item in tqdm(image_dataset, desc="Building image dictionary")}
        raw_datasets = load_dataset(dataset_name[0], dataset_name[1][1])
        raw_dataset = list(raw_datasets.values())[0]
    elif isinstance(dataset_name, HFDataset):
        raw_dataset = dataset_name 
    else:
        raw_datasets = load_dataset(dataset_name)
        # Use only first dict
        raw_dataset = list(raw_datasets.values())[0]
        image_dataset = None

    
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_raw_dataset = split_dataset['train']
    eval_raw_dataset = split_dataset['test']

    train_dataset = VQADataset(train_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=id_to_image)
    valid_dataset = VQADataset(eval_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=id_to_image)
    train_debug = Subset(train_dataset, indices=range(50))
    valid_debug = Subset(valid_dataset, indices=range(50))

    return train_dataset, valid_dataset, train_debug, valid_debug



def get_llava_datasets(dataset_name, image_processor, tokenizer, tokenizer_max_length=128, df_dir=None, img_dir=None):
    df = pd.read_json(df_dir)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    ds = HFDataset.from_pandas(df)

    split_dataset = ds.train_test_split(test_size=0.2, seed=42)
    train_raw_dataset = split_dataset['train']
    eval_raw_dataset = split_dataset['test']

    train_dataset = LlavaInstructDataset(train_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=img_dir)
    valid_dataset = LlavaInstructDataset(eval_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=img_dir)
    train_debug = Subset(train_dataset, indices=range(50))
    valid_debug = Subset(valid_dataset, indices=range(50))

    return train_dataset, valid_dataset, train_debug, valid_debug



def get_aok_datasets(dataset_name, image_processor, tokenizer, tokenizer_max_length=64, img_dir=None):
    
    raw_datasets = load_dataset(dataset_name)
    raw_dataset = list(raw_datasets.values())[0]

    
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_raw_dataset = split_dataset['train']
    eval_raw_dataset = split_dataset['test']

    train_dataset = AOKVQADataset(train_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=id_to_image)
    valid_dataset = AOKVQADataset(eval_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=id_to_image)
    train_debug = Subset(train_dataset, indices=range(50))
    valid_debug = Subset(valid_dataset, indices=range(50))

    return train_dataset, valid_dataset, train_debug, valid_debug


def get_visual7w_datasets(dataset_name, image_processor, tokenizer, tokenizer_max_length=64, img_dir=None):
    
    id_to_image = None
    
    raw_datasets = load_dataset(dataset_name)
    # Use only first dict
    raw_dataset = list(raw_datasets.values())[0]

    flattened_data = []
    for example in raw_dataset:
        flattened_data.extend(flatten_qa(example))  

    flat_dataset = HFDataset.from_list(flattened_data)

    
    split_dataset = flat_dataset.train_test_split(test_size=0.2, seed=42)
    train_raw_dataset = split_dataset['train']
    eval_raw_dataset = split_dataset['test']

    train_dataset = Visual7wDataset(train_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=id_to_image)
    valid_dataset = Visual7wDataset(eval_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=id_to_image)
    train_debug = Subset(train_dataset, indices=range(50))
    valid_debug = Subset(valid_dataset, indices=range(50))

    return train_dataset, valid_dataset, train_debug, valid_debug




def export_qna_from_conversation(item, seed=42):
    conv = item['conversations']
    if not conv:
        return None
    qna_list = []
    for i in range(0, len(conv) - 1, 2):
            q, a = conv[i], conv[i + 1]
            if q.get("from") == "human" and a.get("from") == "gpt":
                question = re.sub(r"<image>\s*", "", q["value"]).strip()
                answer   = a["value"].strip()
                if len(answer) < 12:
                    continue
                if "[" in question and "]" in question:
                    continue
                if "[" in answer and "]" in answer:
                    continue

                qna_list.append([question, answer])

    if not qna_list:
        return None
                

    local_random = random.Random(seed)

    qna = local_random.choice(qna_list)
    
    return qna[0], qna[1]




def flatten_qa(example):
    script_prefix = ["You are a helpful AI that answers multiple-choice questions based on the given image.\n"
                "Select only the single best answer from A, B, C, or D.\n\n"
                "Respond only with one letter (e.g., 'A').\n\n"]
    script_suffix = ["Answer:"]
    dialog = example["data"]
    image = example['images']
    flat_rows = []

    for i in range(1, len(dialog), 2):
        question_script, answer_script = dialog[i]['data'], dialog[i+1]['data']
        question = '\n'.join(script_prefix+question_script.split('\n')[:-1]+script_suffix)
        choices = [line.strip()[:-1] for line in question_script.split('\n') if line[:2] in ['A.', 'B.', 'C.', 'D.']]
        answer = answer_script[-1] # Get Answer Letter from the script
        index = ord(answer) - ord('A')
        answer = choices[index]

        flat_rows.append({
            "image": image,
            "question": question,
            "answer": answer
        })
    
    return flat_rows



def contains_chinese(text):
    return any('\u4e00' <= c <= '\u9fff' for c in text)
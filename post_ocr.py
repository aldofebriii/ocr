from datasets import load_dataset
from exception import BadRequestException
import numpy as np
from PIL import Image
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
import torch
import itertools
import math
import re
from dataclasses import dataclass, field, asdict
from typing import List
import time

MENU_NM = 'menu.nm'
MENU_CNT = 'menu.cnt'
MENU_UNIT_PRICE = 'menu.unitprice'
MENU_PRICE = 'menu.price'
SERVICE_CHARGE = 'sub_total.service_price'
TAX_CHARGE = 'sub_total.tax_price'
DISCOUNT_PRICE = 'sub_total.discount_price'
OTHER_CHARGE = 'sub_total.othersvc_price'

@dataclass
class Item:
    data: dict
    level: int

@dataclass
class Result:
    items: List[Item] = field(default_factory=list)
    charge: dict = field(default_factory=dict)
    time: int = field(default_factory=int)


class PostOcr:
    def __init__(self):
        datasets = load_dataset("katanaml/cord");
        self.labels: list[str] = datasets['train'].features['ner_tags'].feature.names
        self.processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LayoutLMv2ForTokenClassification.from_pretrained("katanaml/layoutlmv2-finetuned-cord")
        self.id2label = self.model.config.id2label
        self.model.to(device)
        self.patterns = self.__generate_pattern()
    
    def __parsing_price(self, price: str):
        trimmed_price = price.strip()
        clean_price = trimmed_price
        comma_exists = trimmed_price.find(',') > - 1
        dot_exists = trimmed_price.find('.') > -1
        each_exists = trimmed_price.find('@') > -1
        rp_pattern = r"(?i)Rp"
        rp_exists = re.search(pattern=rp_pattern, string=trimmed_price)
        
        if(each_exists):
            clean_price = trimmed_price.replace('@', '')
        if(rp_exists):
            clean_price = re.sub(pattern=rp_pattern, repl='', string=trimmed_price).strip()
        
        if(comma_exists):
            clean_price = clean_price.replace(",", "")
        
        if(dot_exists and not comma_exists):
            clean_price = clean_price.replace('.', '')
        try:
            price_clean = float(clean_price)
            return price_clean if not math.isnan(price_clean) else 0
        except ValueError:
            return 0
        
    def __generate_pattern(self):
        # Define groups
        self.group_1 = [MENU_NM, MENU_CNT]
        self.group_2 = [MENU_UNIT_PRICE, MENU_PRICE]
        self.group_3 = [MENU_CNT, MENU_PRICE] #NEED TO BE VALIDATE USING REGEX
        self.group_4 = [MENU_CNT, MENU_UNIT_PRICE] #NEED TO BE VALIDATE USING REGEX
        self.group_5 = [MENU_NM, MENU_PRICE] #OPTIONAL LEVEL TO LEVEL 3
        self.group_6 = [MENU_NM, MENU_UNIT_PRICE] #OPTIONAL LEVEL TO LEVEL 3 using regex

        # Generate all possible permutations for group_1 and group_2
        group_1_permutations = list(itertools.permutations(self.group_1, 2))
        group_2_permutations = list(itertools.permutations(self.group_2, 2))

        # Combine them to form the full arrays
        patterns = [
            list(g1) + list(g2)              # Combine permutations from both groups
            for g1 in group_1_permutations
            for g2 in group_2_permutations
        ] + [
            list(g1) + [MENU_PRICE]          # Add a single MENU_PRICE to g1
            for g1 in group_1_permutations
        ] + [
            list(g1) + [MENU_UNIT_PRICE]     # Add a single MENU_UNIT_PRICE to g1
            for g1 in group_1_permutations
        ] + [
            [MENU_NM] + list(g2)
            for g2 in group_2_permutations
        ] + [
            [MENU_NM] + [MENU_PRICE],        # Add specific combinations
        ] + [
            [MENU_NM] + [MENU_UNIT_PRICE]
        ] + [
            self.group_3 #Wrong Parsing need to be validate
        ] + [
            self.group_4
        ]
        return patterns

    def process(self, image: Image, bbox: list[list[int]], words: list[str]):
        if len(bbox) != len(words):
            raise BadRequestException("invalid bbox and words")
        image = image.convert("RGB")
        start_time = time.time()
        encoded_inputs = self.processor(image, words, boxes=bbox, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = encoded_inputs.pop('offset_mapping')
        for k,v in encoded_inputs.items():
            encoded_inputs[k] = v.to(self.device)
        outputs = self.model(**encoded_inputs)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoded_inputs.bbox.squeeze().tolist()
        width, height = image.size
        is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
        true_predictions = [self.id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
        predicted_boxes = [[box, width, height] for idx, box in enumerate(token_boxes) if not is_subword[idx]]
        classified_words = []
        raw_classified_order = []
        for index, (t) in enumerate(predicted_boxes):
            box = t[0]
            if box in bbox:
                word_index = bbox.index(box)
                classified_words .append((words[word_index]))
                raw_classified_order.append(true_predictions[index][2:]) 
        duplicated_raw = raw_classified_order.copy()
        duplicated_words = classified_words.copy()
        result = Result()
        print(duplicated_raw, duplicated_words)
        # Get the length of the target subsequence
        for pattern in self.patterns:
            target_length = len(pattern) 
            level = target_length   
            i = 0
            while i < (len(duplicated_raw) - target_length + 1):
                subarray = duplicated_raw[i:i + target_length]
                # Check if the current window matches the target
                if subarray == pattern:
                    data = {}
                    for j in range(target_length):
                        label = duplicated_raw[i + j]
                        word = duplicated_words[i + j]
                        regexValidation = False
                        if((
                            (pattern == self.group_3 or pattern == self.group_4) and label == MENU_CNT) or
                            (pattern == self.group_5 or pattern == self.group_6) and label == MENU_NM
                        ):
                            regex = r"(\d+)\s*(.+)" #REGEX VALIDATION
                            match = re.match(pattern=regex, string=word)
                            if match:
                                #Match the mis classified
                                label = MENU_NM
                                data[MENU_CNT] = int(match.group(1))
                                if match.group(2) == '' :
                                    continue
                                word = match.group(2)
                                level = 3 # 3 Level for Count, Name and Price
                                regexValidation = True
                        if label == MENU_PRICE or label == MENU_UNIT_PRICE:
                            parsed_price = self.__parsing_price(word)
                            word = parsed_price
                        if label == MENU_CNT and type(word) == str and not regexValidation:
                            match = re.search(r'\d+', word)
                            if(match):
                                word = int(match.group())
                            else:
                                word = 1
                        data[label] = word
                    map_classified = Item(data, level)
                    print(pattern, map_classified)
                    result.items.append(map_classified)
                    del duplicated_words[i:i + target_length] 
                    del duplicated_raw[i:i + target_length]
                else:
                    i+=1
        #try to find the service and tax charge
        j = 0
        while(j < len(duplicated_raw)):
            row = duplicated_raw[j]
            if(row in [SERVICE_CHARGE, TAX_CHARGE, DISCOUNT_PRICE, OTHER_CHARGE]):
                parsed_price = self.__parsing_price(duplicated_words[j])
                field_name = ''
                if(row == SERVICE_CHARGE):
                    field_name = 'service_charge'
                if(row == TAX_CHARGE):
                    field_name = 'tax_charge'
                if(row == DISCOUNT_PRICE):
                    field_name = 'discount_charge'
                if(row == OTHER_CHARGE):
                    field_name = 'other_charge'
                if parsed_price >= 0:
                    print(field_name, parsed_price)
                    result.charge[field_name] = parsed_price
            j+=1
        end_time = time.time()
        result.time = end_time - start_time
        return asdict(result)
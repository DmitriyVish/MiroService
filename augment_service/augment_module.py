import torch
torch.cuda.is_available = lambda: False
import logging
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskGenerator:
    """
    Этот класс отвечает за:
    1. Загрузку выбранной русскоязычной GPT-модели (ruGPT3XL, ruGPT3Small и т.д.)
    2. Токенизацию входного текста (промпта)
    3. Вызов метода генерации модели
    4. Декодирование результата в читаемый текст
    """
    # Словарь поддерживаемых моделей и их путей на Hugging Face Hub    
    SUPPORTED_MODELS = {
        "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2-1.5B": "Qwen/Qwen2-1.5B",
        "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
        "Qwen2-0.5B": "Qwen/Qwen2-0.5B",
        "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507"
    }
    
    def __init__(self, model_name: str, device: str = None):
        """
        Инициализирует экземпляр TaskGenerator.
        Args:
            model_name (str): Имя модели, например 'ruGPT3Small'.
            device (str, optional): Устройство ('cpu', 'cuda', 'cuda:0'). Если None, выбирается автоматически.
        """
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f" Model {model_name} is not supported. "
                             f"Choose from {list(self.SUPPORTED_MODELS.keys())}")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.SUPPORTED_MODELS[model_name]
        
        logger.info(f"Loading model {self.model_path} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        except Exception as e:
            logger.warning(f"Failed with Auto classes. Error {e}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path) 
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token 
            
        self.model.to(self.device)      
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def generate_tasks(self, prompt: str, num_return_sequences: int = 5, max_length: int = 200, **kwargs) -> list[str]:
        """
        Генерирует несколько вариантов текста на основе промпта.
        Args:
            prompt (str): Входной текст.
            num_return_sequences (int): Сколько вариантов сгенерировать.
            max_length (int): Максимальная длина одного сгенерированного текста.
            **kwargs: Дополнительные параметры для генерации (temperature, top_p и т.д.).
        Returns:
            list[str]: Список сгенерированных текстов (без промпта).
        """
        # Токенизация промпта       
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt}
        ]

        # Шаблон чата токенизатора        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Токенизация
        inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = inputs.to(self.device)

        # attention_mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()

        gen_kwargs = kwargs.copy() 
        temperature = gen_kwargs.pop('temperature', 0.7) # Извлекаем и удаляем из gen_kwargs
        top_p = gen_kwargs.pop('top_p', 0.9)            # Извлекаем и удаляем из gen_kwargs
        
        # Генерируем текст
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_length, 
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=attention_mask,              
                temperature=temperature,
                top_p=top_p,
                do_sample=True,                
                **gen_kwargs
            )

        # Декодирование и извлечение
        generated_texts = []
        for output in outputs:
            decoded_text = self.tokenizer.decode(output, skip_special_tokens=True)

            # --- ОТЛАДКА ---
            print("--- DEBUG INFO ---")
            print(f"Full prompt with assistant start:\n{repr(text)}") # repr показывает спецсимволы
            print(f"Decoded text:\n{repr(decoded_text)}")
            print(f"Length of full prompt: {len(text)}")
            print(f"Length of decoded text: {len(decoded_text)}")
            # Проверим, есть ли маркеры
            im_end_pos = decoded_text.find("<|im_end|>")
            eos_pos = decoded_text.find(self.tokenizer.eos_token) if self.tokenizer.eos_token else -1
            print(f"Position of <|im_end|>: {im_end_pos}")
            print(f"Position of EOS token ({self.tokenizer.eos_token}): {eos_pos}")
            print("--- END DEBUG ---")

            # --- ОБРЕЗКА ---            
            full_prompt_with_assistant_start = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )           
            start_idx = len(full_prompt_with_assistant_start)            
            generated_part_full = decoded_text[start_idx:].strip()           
            end_marker_im_end = generated_part_full.find("<|im_end|>")
            end_marker_eos = generated_part_full.find(self.tokenizer.eos_token) if self.tokenizer.eos_token else -1           
            end_indices = [idx for idx in [end_marker_im_end, end_marker_eos] if idx != -1]
            min_end_idx = min(end_indices) if end_indices else len(generated_part_full)            
            generated_part_clean = generated_part_full[:min_end_idx].strip()

            print(f"Extracted generated part: {repr(generated_part_clean)}") # Отладка извлеченной части
            print("------------------")

            generated_texts.append(generated_part_clean)

        logger.info(f"Generated {len(generated_texts)} tasks.")
        return generated_texts

       


       



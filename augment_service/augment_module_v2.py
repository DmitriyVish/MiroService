import torch
import logging
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel

# Отключаем CUDA (для примера; уберите, если нужно GPU)
torch.cuda.is_available = lambda: False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class TaskGenerator:
    """
    Класс для генерации текста на основе промпта с использованием языковых моделей.
    Функции:
    1. Загрузка модели из списка поддерживаемых.
    2. Токенизация входного текста.
    3. Генерация текста моделью.
    4. Декодирование и очистка результата.
    """

    # Список поддерживаемых моделей (добавлены русскоязычные варианты)
    SUPPORTED_MODELS = {
        "ruGPT3Small": "sberbank-ai/rugpt3small_based_on_gpt2",
        "ruGPT3Medium": "sberbank-ai/rugpt3medium_based_on_gpt2",
        "ruGPT3Large": "sberbank-ai/rugpt3large_based_on_gpt2",
        "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2-1.5B": "Qwen/Qwen2-1.5B",
        "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
        "Qwen2-0.5B": "Qwen/Qwen2-0.5B",
        "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507"
    }

    def __init__(self, model_name: str, device: str = None):
        """
        Инициализация генератора.
        Args:
            model_name (str): Имя модели из SUPPORTED_MODELS.
            device (str, optional): Устройство ('cpu', 'cuda', 'cuda:0'). 
                                 Если None — выбирается автоматически.
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Модель {model_name} не поддерживается. "
                f"Выберите из: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.SUPPORTED_MODELS[model_name]

        logger.info(f"Загрузка модели {self.model_path} на {self.device}...")

        # Попытка загрузки через Auto-классы
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        except Exception as e:
            logger.warning(f"Ошибка при загрузке через Auto-классы: {e}. "
                         "Пробуем через GPT2-классы...")
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            except Exception as e2:
                raise RuntimeError(
                    f"Не удалось загрузить модель {self.model_path}: {e2}"
                )

        # Установка pad_token, если отсутствует
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Установлен pad_token = eos_token")
            else:
                raise ValueError("Не удалось определить pad_token для токенизатора")

        # Перенос модели на устройство и режим inference
        self.model.to(self.device)
        self.model.eval()
        logger.info("Модель успешно загружена")

    def generate_tasks(
        self,
        prompt: str,
        num_return_sequences: int = 5,
        max_length: int = 200,
        **kwargs
    ) -> list[str]:
        """
        Генерирует несколько вариантов текста на основе промпта.
        Args:
            prompt (str): Входной текст-промпт.
            num_return_sequences (int): Количество генерируемых вариантов.
            max_length (int): Максимальная длина одного сгенерированного текста.
            **kwargs: Дополнительные параметры генерации 
                        (temperature, top_p, top_k и др.).
        Returns:
            list[str]: Список сгенерированных текстов (без промпта).
        """
        # Формирование чата для токенизации
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Токенизация промпта с шаблоном чата
        try:
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.error(f"Ошибка при формировании чата: {e}")
            full_prompt = prompt  # fallback

        # Токенизация
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # attention_mask (если pad_token определён)
        attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else None

        # Параметры генерации
        gen_kwargs = kwargs.copy()
        allowed_gen_params = {
            'temperature', 'top_p', 'top_k', 'repetition_penalty',
            'num_beams', 'early_stopping', 'no_repeat_ngram_size'
        }
        gen_kwargs = {
            k: v for k, v in gen_kwargs.items()
            if k in allowed_gen_params
        }

        temperature = gen_kwargs.pop('temperature', 0.7)
        top_p = gen_kwargs.pop('top_p', 0.9)

        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **gen_kwargs
            )

        # Декодирование и постобработка
        generated_texts = []
        for output in outputs:
            decoded_text = self.tokenizer.decode(
                output,
                skip_special_tokens=False,  # Сохраняем спецтокены для обрезки
                clean_up_tokenization_spaces=False
            )

            # Обрезка промпта
            start_idx = len(full_prompt)
            if len(decoded_text) <= start_idx:
                logger.warning("Сгенерированный текст короче промпта")
                generated_part = ""
            else:
                generated_part = decoded_text[start_idx:].strip()

            # Поиск стоп-маркеров
            stop_markers = []
            if "<|im_end|>" in generated_part:
                stop_markers.append(generated_part.find("<|im_end|>"))
            if (self.tokenizer.eos_token and
                    self.tokenizer.eos_token in generated_part):
                stop_markers.append(
                    generated_part.find(self.tokenizer.eos_token)
                )

            # Обрезка по первому стоп-маркеру
            if stop_markers:
                min_stop = min(stop_markers)
                generated_part = generated_part[:min_stop].strip()

            generated_texts.append(generated_part)

        logger.info(f"Сгенерировано {len(generated_texts)} вариантов")
        return generated_texts
# Submission Guide / Руководство по отправке / Тапсыру нұсқаулығы

## English

### What to Submit

Your submission is a **directory** containing any combination of:

1. **LoRA adapter weights** (optional) — the output of `model.save_pretrained()` from PEFT:
   - `adapter_config.json`
   - `adapter_model.safetensors` (or `adapter_model.bin`)

2. **Custom generation code** (optional) — a file named `generate.py` with a function:
   ```python
   def generate(model, tokenizer, prompt: str, format_name: str) -> str:
       """Generate structured output for the given prompt."""
       ...
   ```

### what happens if you omit something

YOU ARE NOT REQUIRED TO SUBMIT BOTH - You can either submit one part of solution or both

| LoRA weights | generate.py | What happens |
|:---:|:---:|---|
| ✓ | ✓ | Your LoRA is merged, your generate function is used |
| ✓ | ✗ | Your LoRA is merged, default generation is used |
| ✗ | ✓ | Base model is used, your generate function is used |
| ✗ | ✗ | Base model + default generation (baseline score) |

### Constraints

- **Max adapter size**: 100 MB
- **Generation timeout**: 5 seconds per sample
- **No internet access** during evaluation

### Valiwate before submitting

```bash
python -m submission.validate_submission path/to/your/submission
```

---

## Русский

### Что нужно отправить

Отправка — это **директория**, содержащая любую комбинацию:

1. **Веса LoRA адаптера** (опционально) — результат `model.save_pretrained()` из PEFT:
   - `adapter_config.json`
   - `adapter_model.safetensors` (или `adapter_model.bin`)

2. **Код генерации** (опционально) — файл `generate.py` с функцией:
   ```python
   def generate(model, tokenizer, prompt: str, format_name: str) -> str:
       """Генерирует структурированный вывод для заданного промпта."""
       ...
   ```

### Что происходит при пропуске компонентов

ВАМ НЕОБЯЗАТЕЛЬНО ОТПРАВЛЯТЬ ОБА КОМПОНЕНТА - Вы можете использовать только один, или оба сразу

| Веса LoRA | generate.py | Результат |
|:---:|:---:|---|
| ✓ | ✓ | Ваш LoRA объединяется с моделью, используется ваша функция генерации |
| ✓ | ✗ | Ваш LoRA объединяется с моделью, используется стандартная генерация |
| ✗ | ✓ | Базовая модель, используется ваша функция генерации |
| ✗ | ✗ | Базовая модель + стандартная генерация (базовый балл) |

### Ограничения

- **Макс. размер адаптера**: 100 МБ
- **Тайм-аут генерации**: 5 секунд на пример
- **Без доступа к интернету** во время оценки

### Проверка перед отправкой

```bash
python -m submission.validate_submission путь/к/вашей/отправке
```

---

## Қазақша

### Не тапсыру керек

Тапсыру — кез келген комбинацияны қамтитын **директория**:

1. **LoRA адаптер салмақтары** (міндетті емес) — PEFT-тегі `model.save_pretrained()` нәтижесі:
   - `adapter_config.json`
   - `adapter_model.safetensors` (немесе `adapter_model.bin`)

2. **Генерация коды** (міндетті емес) — функциясы бар `generate.py` файлы:
   ```python
   def generate(model, tokenizer, prompt: str, format_name: str) -> str:
       """Берілген сұрау үшін құрылымдық шығысты генерациялайды."""
       ...
   ```

### Компоненттерді өткізіп жіберу кезінде не болады

| LoRA салмақтары | generate.py | Нәтиже |
|:---:|:---:|---|
| ✓ | ✓ | Сіздің LoRA біріктіріледі, сіздің генерация функциясы қолданылады |
| ✓ | ✗ | Сіздің LoRA біріктіріледі, стандартты генерация қолданылады |
| ✗ | ✓ | Базалық модель, сіздің генерация функциясы қолданылады |
| ✗ | ✗ | Базалық модель + стандартты генерация (базалық балл) |

### Шектеулер

- **Адаптердің макс. өлшемі**: 100 МБ
- **Генерация тайм-ауты**: бір мысалға 5 секунд
- **Бағалау кезінде интернетке рұқсат жоқ**

### Тапсыру алдында тексеру

```bash
python -m submission.validate_submission тапсыру/директориясының/жолы
```

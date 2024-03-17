# Как работают текущие текстовые диффузии?

Сейчас надо сделать бейзлайн, один из вариантов - отталкиваться от существующих тексотвых диффузий. Вадик предложил позапускать их и посмотреть, как оно работает с точки зрения скорости. Логика такая - `генерирует за мало шагов -> сравнительно простая модель -> должна обучиться без титанических усилий`.


Список (там, где обещают чекпоинты, судя по README):
    - [difformer](https://github.com/zhjgao/difformer/tree/main)
    - [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq?tab=readme-ov-file)
    - [Ещё что-то](https://github.com/yegcjs/DiffusionLLM)
    - [Plaid](https://github.com/igul222/plaid)
    - [Bridge_Gap_Diffusion](https://github.com/ZetangForward/Bridge_Gap_Diffusion)
    - [ssd-lm](https://github.com/xhan77/ssd-lm?tab=readme-ov-file)


## difformer

Запускал в лабном докере - `registry.kontur.host/library/srs/research/lab-miniconda3-cudnn8-cuda11.3_extended:20230601`

Через `pip install -e .` fairseq не ставился, ошибка как [тут](https://github.com/facebookresearch/fairseq/issues/3535). Поставил его отдельно. Потом была ещё ошибка, пришлось ставить python3.8, и с ним работать.
```
git clone git@github.com:zhjgao/difformer.git
wget https://github.com/facebookresearch/fairseq/archive/refs/tags/v0.10.2.zip
unzip v0.10.2.zip 
cd fairseq-0.10.2
pip install -e .
```

Модели и данны пришлось скачать руками и разложить по папочкам.

Изначально стояло 20 шагов диффузии, погенерил так, ещё погенерил на 10 шагах. Инференс довольно быстрый, сильного различия в результах нет

## DiffuSeq

```
git clone git@github.com:Shark-NLP/DiffuSeq.git
gdown --id 1gj9OpGlM9OzbbrCIOfia8Ve6GMDd2Vxa
cd datasets/QQP
gdown --id 1vRWUY-qIuTa4TF1bgL2OxpgmMYQEwoEq
cd ../../scripts/
pip install blobfile
bash scripts/run_decode.sh  
```

Запустил генерацию на 200, 1000, 2000 степов - результаты смотреть в папках `generation_outputs_*_steps`


# Plaid

Нужна A100 для flash-attention

Ставим под cuda 12.1 т.к такая в текущем контейнере, а мне лень собирать под каждую модель свой контейнер.
Из requirements.txt удаляем торч

```
git clone git@github.com:igul222/plaid.git
pip install -r requirements.txt
pip install https://download.pytorch.org/whl/cu121/torch-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl
git clone https://github.com/HazyResearch/flash-attention.git
```

Попробовал:
```
pip install ./flash-attention
```

Получил:
```
 Building wheel for flash-attn (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [9 lines of output]
      
      
      torch.__version__  = 2.1.0+cu121
      
      
      running bdist_wheel
```

Попробовал:
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```


```
pip install ./flash-attention/csrc/rotary
pip install ./flash-attention/csrc/fused_dense_lib
```

Скачиваем чекпоинт:
```
cd plaid
wget https://github.com/igul222/plaid/releases/download/v1.0.0/plaid1b_weights.tar.gz.00
wget https://github.com/igul222/plaid/releases/download/v1.0.0/plaid1b_weights.tar.gz.01
wget https://github.com/igul222/plaid/releases/download/v1.0.0/plaid1b_weights.tar.gz.02
python sample.py --weights_path=/diploma/diplom/exps/current_text_diffusion_research/plaid/plaid1b_weights --dim=2048 --n_blocks=24 --n_heads=32 --seq_len=1024
```

# ssd-lm

Там есть нотебучек, прямо в колабе запустить можно.

Там дефолтный семплер - cosine, с ним при генерации на 1000 степов что-то адекватное начинало получаться при t=400 т.е при 600 степах.
Ещё попробовал LMSDiscreteScheduler и EulerDiscreteScheduler, но получается бред какой-то
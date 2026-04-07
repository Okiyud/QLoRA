# Fine-tuning de um modelo utilizando QLoRA

Este repositório contém um trabalho de Processamento de Linguagem Natural. O trabalho consiste no fine-tuning de um Small Language Model (SML) com um dataset sintético produzido por um Large Language Model (LLM). O dataset deve conter informações sobre um assunto atual (2025/2026) para avaliar o aprendizado do modelo por meio do treinamento utilizando QLoRA.

O tema escolhido foi o lançamento do álbum do Gorillaz, "The Mountain". O álbum foi lançado em 27 de Fevereiro de 2026. 

![image](image.png)

## Produção do dataset sintético

Os dados iniciais foram produzidos utilizando conteúdo disponível da internet. Os textos foram retirados da Wikipedia e do site Genius (um site de música). Os textos extraídos são passados como contexto para o LLM produzir pares de instrução e resposta.

Referências:
- https://en.wikipedia.org/wiki/The_Mountain_(Gorillaz_album). Acessado em 26/03/2026

- https://genius.com/albums/Gorillaz/The-mountain. Acessado em 26/03/2026


## Bibliotecas e versões

O código foi realizado na linguagem Python. Como boa parte do uso do SLM foi feita no ambiente do Google Colab, a versão do python recomendada é a versão usada durante a execução deste trabalho, versão `3.12.13`.

Recomenda-se o uso da mesma versão python para evitar incompatibilidade com as bibliotecas.

```python
pip install torch==2.x.x --index-url https://download.pytorch.org/whl/cu128
```

```python
pip install transformers==5.5.0 datasets==3.3.2 accelerate==1.4.0 bitsandbytes==0.49.2 trl==0.15.2 peft==0.14.0 gdown python-dotenv google-genai jupyter ipykernel
```

## Estrutura do repositório

- `base_conhecimento/`: Diretório que contém os textos extraídos em forma de blocos (arquivos). Cada arquivo markdown neste diretório está relacionado a um subtema dos textos.
- `prompt/`: Diretório com os prompts usados para criação dos dados sintéticos.
- `gerador_dados_sintéticos.py`: Script python para criação dos dados sintéticos. O código faz checkpoints para evitar perda e salva tudo num diretório `data/`. Utiliza API do Gemini para criação, por isso é necessário colocar sua própria chave em `.env`.
- `Tuning_Qwen3.ipynb`: Jupyter notebook que contém o treinamento QLoRA do modelo Qwen3. Ele armazena os adaptadores no diretório `final-adapter/`.
- `Chat_Qwen3.ipynb`: Jupyter notebook que faz o teste do modelo Qwen3 antes e depois do fine-tuning, utilizando 10 prompts diferentes para avaliação.

OBS: os diretórios `data/` (contendo os dados sintéticos) e `final-adapter/` (os adaptadores QLoRA) **NÃO** estão neste repositório por conta do tamanho. Porém, estão disponíveis nos links abaixo. Não é necessário transferí-los para este repositório caso queira usar os dados obtidos na minha iteração, pois os dois Jupyter Notebooks fazem o download automático caso não detectem os diretórios necessários.

- `data/`: https://drive.google.com/drive/folders/1NwcGXKS81wNPqpbGg2gX8i4WqGTwmYt_?usp=drive_link
- `final-adapter/`: https://drive.google.com/drive/folders/1KFSGdqg5H5MeE8QA7Dt7tjt904_I7bwL?usp=sharing


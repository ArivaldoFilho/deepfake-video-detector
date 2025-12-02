# deepfake-video-detector

Aqui está o **README.md** exatamente no formato Markdown puro, pronto para ser copiado e colocado no repositório GitHub sem ajustes adicionais.

---

# README.md

# Deepfake Video Detector – Sistema Híbrido de Detecção de Manipulações Faciais

Este projeto implementa um sistema híbrido avançado para detecção de deepfakes em imagens e vídeos.
A solução combina três abordagens distintas:

1. Um modelo CNN treinado diretamente em imagens do dataset RVF10K.
2. Um modelo determinístico baseado em extração manual de características (LBP, FFT, Laplacian, detecção de bordas, intensidade etc.).
3. Um meta-modelo de Stacking que combina as previsões dos dois modelos anteriores para obter maior robustez.

A aplicação final inclui uma interface completa em Streamlit capaz de analisar vídeos, detectar rostos, extrair frames e produzir um veredito indicando se o conteúdo é autêntico ou manipulado.

---

## 1. Dataset Utilizado

Dataset utilizado: **RVF10K – Real vs Fake Faces**
Disponível em:
[https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k/data?select=rvf10k](https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k/data?select=rvf10k)

Estrutura utilizada no projeto:

* Total de imagens: 10.000
* Classes: real e fake
* Divisão:

  * 7.000 imagens para treino (3.500 reais, 3.500 falsas)
  * 3.000 imagens para validação (1.500 reais, 1.500 falsas)

---

## 2. Arquitetura da Solução

### 2.1 Modelo CNN

A CNN recebe imagens RGB redimensionadas para 128×128 e processadas da seguinte maneira:

* Normalização 1/255
* Três blocos convolucionais:

  * Conv2D → BatchNormalization → MaxPooling
  * Filtros: 32, 64 e 128
* GlobalAveragePooling2D
* Dropout (0.6)
* Dense(64, ativação ReLU)
* Saída: Dense(1, ativação Sigmoid)

Treinada com Early Stopping monitorando `val_loss`.

---

### 2.2 Modelo Determinístico (Handcrafted Features)

O segundo modelo utiliza extração manual de características (OpenCV e NumPy).

Características extraídas:

1. Detecção de rosto via Haar Cascade
2. Estatísticas de intensidade

   * Média
   * Desvio padrão
3. Laplacian variance (nitidez/blur)
4. Edge density (Canny)
5. Frequência FFT (high-frequency ratio)
6. LBP manual com histograma de 256 bins

Total: **261 features por imagem**

Algoritmo de classificação:

* StandardScaler
* Logistic Regression (max_iter = 2000)

---

### 2.3 Meta-Modelo de Stacking

O sistema híbrido combina:

* Probabilidade da CNN
* Probabilidade do modelo determinístico

O meta-modelo Logistic Regression aprende a ponderar as duas previsões, obtendo uma classificação final mais robusta.

Benefícios:

* Menor variância
* Combinação de padrões visuais e estatísticos
* Redução de erros isolados dos modelos independentes

---

## 3. Resultados Obtidos

Resultados do treinamento efetuado sobre o conjunto RVF10K:

### CNN v2

* Acurácia: 0.6363
* Precisão: 0.6423
* Recall: 0.6153
* F1-Score: 0.6285

### Modelo Determinístico v2

* Acurácia: 0.6817
* Precisão: 0.6813
* Recall: 0.6827
* F1-Score: 0.6820

### Modelo Híbrido (Stacking)

* Acurácia: 0.6817
* Precisão: 0.6818
* Recall: 0.6813
* F1-Score: 0.6816

O modelo determinístico demonstrou desempenho consistente, enquanto o stacking aumentou a estabilidade geral das previsões.

---

## 4. Aplicação Streamlit (Detector de Vídeos)

A aplicação Streamlit disponibilizada permite:

* Upload de vídeos (MP4, AVI, MOV)
* Detecção de rosto por frame
* Processamento híbrido (CNN + determinístico + stacking)
* Suavização temporal de probabilidades
* Renderização de vídeo com identificação (Real/Fake)
* Estatísticas de processamento
* Veredito final com confiança média global

A interface apresenta um design voltado para ambientes técnicos, com foco na clareza da análise.

---

## 5. Execução do Projeto

### 5.1 Requisitos

* Python 3.9+
* TensorFlow
* OpenCV
* Scikit-Learn
* Streamlit
* NumPy
* Joblib

### 5.2 Treinar os Modelos

```
python train_hybrid_rvff10k.py
```

Os modelos treinados serão salvos dentro da pasta:

```
models/
```

### 5.3 Executar a Interface Streamlit

```
streamlit run detect_video-v2.py
```

Acesse:

```
http://localhost:8501
```

---

## 6. Considerações Finais

Este projeto demonstra como a combinação de Deep Learning e métodos determinísticos pode resultar em um sistema mais robusto para detecção de deepfakes.
A integração via Stacking equilibra características de alto nível (CNN) e padrões texturais/estatísticos (modelo determinístico), oferecendo maior generalização.

O projeto pode ser expandido para:

* Redes neurais pré-treinadas (EfficientNet, Xception)
* Detecção multimodal (vídeo + áudio)
* Transformers para análise temporal
* Aumento de dados e hard negatives

---


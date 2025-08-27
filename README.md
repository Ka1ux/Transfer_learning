# 🐱🐶 Classificação de Imagens com Transfer Learning (Xception)

Este projeto implementa um modelo de **classificação de imagens** usando **Transfer Learning** com a arquitetura **Xception** pré-treinada no ImageNet.  
O código foi desenvolvido para rodar no **Google Colab**, utilizando imagens armazenadas no **Google Drive**.

---

## 🚀 Funcionalidades
- Monta automaticamente o Google Drive no Colab
- Busca a pasta `minhas_imagens/` em diferentes diretórios possíveis
- Verifica a estrutura de classes automaticamente
- Aplica **Data Augmentation** para melhorar a robustez do modelo
- Utiliza **Xception** como modelo base pré-treinado
- Treina apenas as camadas superiores para classificação binária
- Exibe métricas de acurácia no conjunto de validação

---

## 📂 Estrutura esperada das pastas

Dentro do seu Google Drive, você deve ter uma pasta chamada `minhas_imagens` com **subpastas para cada classe**:

```
minhas_imagens/
 ┣ cats/   # imagens de gatos
 ┗ dogs/   # imagens de cachorros
```

> ⚠️ O modelo foi configurado para **classificação binária**, então são esperadas **duas classes** (exemplo: `cats` e `dogs`).

---

## 🛠️ Tecnologias utilizadas
- [Python 3](https://www.python.org/)
- [Google Colab](https://colab.research.google.com/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [Xception (ImageNet)](https://arxiv.org/abs/1610.02357)

---

## ⚙️ Como usar

1. Abra o código no Google Colab
2. Monte o Google Drive:
   ```python
   drive.mount('/content/drive', force_remount=True)
   ```
3. Coloque suas imagens na pasta `minhas_imagens/` do Drive (com subpastas para cada classe).
4. Execute todas as células do notebook.

---

## 📈 Treinamento do modelo

O script:
- Cria geradores de imagens com `ImageDataGenerator` (com Data Augmentation para treino e apenas normalização para validação).
- Carrega o modelo **Xception** sem a cabeça final (`include_top=False`).
- Congela as camadas convolucionais pré-treinadas.
- Adiciona camadas densas para classificação binária (`sigmoid`).

```python
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
```

Compilação:
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Treinamento:
```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)
```

---

## 📊 Avaliação

Após o treino, o modelo é avaliado no conjunto de validação:
```python
loss, accuracy = model.evaluate(validation_generator)
print(f"Acurácia no conjunto de validação: {accuracy * 100:.2f}%")
```

---

## 📝 Resultados esperados
- Acurácia inicial em torno de **80~90%** dependendo da qualidade das imagens
- Possibilidade de *fine-tuning* descongelando algumas camadas do Xception

---

## 📜 Licença
Este projeto é de uso livre para estudo e pesquisa.  
Sinta-se à vontade para melhorar e adaptar 🚀

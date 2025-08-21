import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import os

# --- Funções Auxiliares para o Grad-CAM (Baseado no tutorial oficial do Keras) ---

def get_img_array(img_path, size):
    """
    Carrega uma imagem do disco e a prepara para o modelo.
    Isso inclui redimensionar e converter para um array numpy.
    """
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    # Adicionamos uma dimensão para que o formato seja (1, height, width, channels)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Gera o mapa de calor (heatmap) para uma imagem de entrada.
    Adaptado para modelos onde a ResNet50 está como submodelo.
    """
    # A ResNet50 está dentro do modelo como subcamada
    resnet_submodel = model.get_layer("resnet50")  # acessa o submodelo

    # Agora pegamos a última camada convolucional dentro da ResNet50
    last_conv_layer = resnet_submodel.get_layer(last_conv_layer_name)

    # Criamos o modelo de gradiente
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # Calcula o gradiente da classe predita
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.6):
    """
    Aplica o heatmap sobre a imagem original e salva o resultado.
    """
    # Carrega a imagem original
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return

    # Redimensiona o heatmap para o tamanho da imagem original
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Converte o heatmap para o formato RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Sobrepõe o heatmap na imagem original
    superimposed_img = heatmap * alpha + img

    # Salva a imagem resultante
    cv2.imwrite(cam_path, superimposed_img)
    print(f"Imagem Grad-CAM salva em: {cam_path}")


# --- BLOCO PRINCIPAL DE EXECUÇÃO ---

if __name__ == '__main__':
    # 1. Carregue seu modelo ResNet50 treinado
    # Certifique-se de que este caminho está correto.
    MODEL_PATH = 'resnet50_colon_model.h5'
    model = keras.models.load_model(MODEL_PATH)
    model.layers[-1].activation = None # Remove a ativação softmax para Grad-CAM

    # 2. Nome da última camada convolucional da ResNet50
    # Você pode verificar isso executando `model.summary()`
    # Normalmente, para a ResNet50, é 'conv5_block3_out'.
    LAST_CONV_LAYER_NAME = "conv5_block3_out"

    # 3. Pré-processamento e tamanho da imagem (deve ser o mesmo do treinamento)
    IMG_SIZE = (224, 224)
    preprocess_input = keras.applications.resnet50.preprocess_input

    # 4. Selecione as imagens que você quer analisar
    # Crie um dicionário com o caminho da imagem e o nome do arquivo de saída
    # DICA: Olhe a matriz de confusão e escolha casos interessantes!
    
    # Exemplo: um pólipo que o modelo provavelmente acertou
    # (Você precisará encontrar um nome de arquivo real na sua pasta de teste)
    caminho_base_teste = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "francismon", "curated-colon-dataset-for-deep-learning", "versions", "1", "test")

    imagens_para_analisar = {
        # CASO 1: Um Pólipo (classe 2) - o modelo deve focar na lesão
        os.path.join(caminho_base_teste, "2_polyps", "test_polyps_ (23).jpg"): "gradcam_polipo_correto.jpg",
        
        # CASO 2: Um caso de Colite (classe 1) - o modelo deve focar na inflamação
        os.path.join(caminho_base_teste, "1_ulcerative_colitis", "test_ulcerative_colitis_ (23).jpg"): "gradcam_colite_correta.jpg",
        
        # CASO 3 (ANÁLISE DE ERRO): Um Pólipo (classe 2) que foi confundido com Normal (classe 0)
        # Procure um caso em que seu modelo errou para uma análise mais profunda!
        # os.path.join(caminho_base_teste, "2_polyps", "NOME_DO_ARQUIVO_QUE_ERROU.jpg"): "gradcam_polipo_falso_negativo.jpg",

        # CASO 4: Um caso Normal (classe 0) - onde o modelo deve focar?
        os.path.join(caminho_base_teste, "0_normal", "test_normal_ (23).jpg"): "gradcam_normal_correto.jpg",
    }
    
    # 5. Gere e salve as imagens Grad-CAM
    for img_path, output_path in imagens_para_analisar.items():
        if not os.path.exists(img_path):
            print(f"[AVISO] Arquivo não encontrado, pulando: {img_path}")
            continue

        # Prepara a imagem para o modelo
        img_array = preprocess_input(get_img_array(img_path, size=IMG_SIZE))

        # Gera o heatmap
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

        # Salva a imagem com o heatmap sobreposto
        save_and_display_gradcam(img_path, heatmap, cam_path=output_path)
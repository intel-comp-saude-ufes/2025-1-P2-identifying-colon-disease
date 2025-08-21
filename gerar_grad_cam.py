import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
import os

# --- Funções Auxiliares para o Grad-CAM (Corrigidas) ---

def get_img_array(img_path, size):
    """
    Carrega uma imagem do disco e a prepara para o modelo.
    """
    try:
        img = keras.utils.load_img(img_path, target_size=size)
        array = keras.utils.img_to_array(img)
        # Adicionamos uma dimensão para que o formato seja (1, height, width, channels)
        array = np.expand_dims(array, axis=0)
        return array
    except Exception as e:
        print(f"Erro ao carregar imagem {img_path}: {e}")
        return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Gera o mapa de calor (heatmap) para uma imagem de entrada.
    """
    # Para modelos Sequential com ResNet50 como primeira camada
    if hasattr(model.layers[0], 'get_layer'):
        base_model = model.layers[0]  # ResNet50 base
        try:
            last_conv_layer = base_model.get_layer(last_conv_layer_name)
            print(f"Camada '{last_conv_layer_name}' encontrada no modelo base.")
        except ValueError:
            print(f"Camada '{last_conv_layer_name}' não encontrada no modelo base.")
            print("Camadas convolucionais disponíveis:")
            for layer in base_model.layers:
                if 'conv' in layer.name.lower():
                    print(f"  {layer.name}")
            return None
    else:
        # Modelo direto
        try:
            last_conv_layer = model.get_layer(last_conv_layer_name)
        except ValueError:
            print(f"Camada '{last_conv_layer_name}' não encontrada no modelo.")
            return None

    # Calcula o gradiente usando uma abordagem mais direta
    with tf.GradientTape() as tape:
        # Faz o forward pass completo
        preds = model(img_array)
        
        # Se não especificou a classe, usa a classe predita
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Pega o score da classe específica
        class_channel = preds[:, pred_index]

    # Pega as ativações da última camada convolucional
    # Criamos um modelo temporário só para isso
    if hasattr(model.layers[0], 'get_layer'):
        # Para modelos Sequential
        base_model = model.layers[0]
        conv_model = keras.models.Model(
            inputs=base_model.input,
            outputs=last_conv_layer.output
        )
        last_conv_layer_output = conv_model(img_array)
    else:
        # Para modelos diretos
        conv_model = keras.models.Model(
            inputs=model.inputs,
            outputs=last_conv_layer.output
        )
        last_conv_layer_output = conv_model(img_array)

    # Agora calculamos os gradientes corretamente
    with tf.GradientTape() as tape:
        # Monitora as ativações da camada convolucional
        tape.watch(last_conv_layer_output)
        
        # Para modelos Sequential, precisamos passar as ativações pelas camadas restantes
        if hasattr(model.layers[0], 'get_layer'):
            # Pega as camadas depois do ResNet50 (flatten, dense, etc.)
            x = last_conv_layer_output
            for layer in model.layers[1:]:  # Pula a primeira camada (ResNet50)
                x = layer(x)
            preds = x
        else:
            # Para modelos não-Sequential, seria diferente
            preds = model(img_array)
        
        # Pega o score da classe
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calcula os gradientes
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Verifica se o gradiente foi calculado corretamente
    if grads is None:
        print("Erro: Gradiente não foi calculado. Verifique se o modelo está configurado corretamente.")
        return None

    # Pooling dos gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiplica cada canal pelo sua importância
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normaliza o heatmap entre 0 e 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Aplica o heatmap sobre a imagem original e salva o resultado.
    """
    # Carrega a imagem original
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return

    # Redimensiona o heatmap para o tamanho da imagem original
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Converte o heatmap para o formato de cor
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    # Converte a imagem original para RGB se necessário
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Sobrepõe o heatmap na imagem original
    superimposed_img = heatmap_rgb * alpha + img_rgb * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Salva a imagem resultante (convertendo de volta para BGR para o OpenCV)
    superimposed_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(cam_path, superimposed_bgr)
    print(f"Imagem Grad-CAM salva em: {cam_path}")


def create_resnet_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Cria o modelo ResNet50 para classificação de colonoscopia.
    """
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Transfer learning (congela a base)
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Melhor que Flatten para ResNet
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def find_last_conv_layer(model):
    """
    Encontra automaticamente a última camada convolucional do modelo.
    """
    conv_layers = []
    
    # Se é um modelo Sequential com ResNet50 como primeira camada
    if hasattr(model.layers[0], 'layers'):
        base_model = model.layers[0]  # ResNet50
        for layer in base_model.layers:
            if 'conv' in layer.name.lower():
                conv_layers.append(layer.name)  # Só o nome, sem prefixo
    else:
        # Modelo direto
        for layer in model.layers:
            if 'conv' in layer.name.lower():
                conv_layers.append(layer.name)
    
    if conv_layers:
        print(f"Camadas convolucionais encontradas: {len(conv_layers)}")
        print(f"Última camada convolucional: {conv_layers[-1]}")
        return conv_layers[-1]
    else:
        print("Nenhuma camada convolucional encontrada!")
        return None

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---

if __name__ == '__main__':
    # 1. Carregue seu modelo ResNet50 treinado
    MODEL_PATH = 'resnet50_colon_model.h5'
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Modelo carregado com sucesso!")
        
        # Remove a ativação softmax da última camada para melhor Grad-CAM
        if hasattr(model.layers[-1], 'activation'):
            model.layers[-1].activation = None
            print("Ativação softmax removida da última camada.")
            
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        print("Criando um modelo de exemplo...")
        model = create_resnet_model()

def debug_model_structure(model):
    """
    Debug: Imprime a estrutura detalhada do modelo para identificar camadas.
    """
    print("\n=== ESTRUTURA DO MODELO ===")
    print(f"Tipo do modelo: {type(model).__name__}")
    print(f"Número de camadas principais: {len(model.layers)}")
    
    for i, layer in enumerate(model.layers):
        print(f"\nCamada {i}: {layer.name} ({type(layer).__name__})")
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            print(f"  -> Sub-modelo com {len(layer.layers)} camadas")
            # Mostra apenas algumas camadas convolucionais importantes
            conv_count = 0
            for j, sublayer in enumerate(layer.layers):
                if 'conv' in sublayer.name.lower():
                    conv_count += 1
                    if conv_count <= 5 or j >= len(layer.layers) - 5:  # Primeiras 5 e últimas 5
                        print(f"    {j}: {sublayer.name}")
                    elif conv_count == 6:
                        print("    ... (camadas intermediárias omitidas)")
            print(f"  -> Total de camadas conv: {conv_count}")
    print("==============================\n")

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---

if __name__ == '__main__':
    # 1. Carregue seu modelo ResNet50 treinado
    MODEL_PATH = 'resnet50_colon_model.h5'
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Modelo carregado com sucesso!")
        
        # DEBUG: Mostra a estrutura do modelo
        debug_model_structure(model)
        
        # Remove a ativação softmax da última camada para melhor Grad-CAM
        if hasattr(model.layers[-1], 'activation'):
            model.layers[-1].activation = None
            print("Ativação softmax removida da última camada.")
            
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        print("Criando um modelo de exemplo...")
        model = create_resnet_model()

    # 2. Encontra automaticamente a última camada convolucional
    LAST_CONV_LAYER_NAME = find_last_conv_layer(model)
    
    # Para ResNet50, as opções mais comuns são:
    if LAST_CONV_LAYER_NAME is None:
        # Opções comuns para ResNet50
        opcoes_camadas = [
            "conv5_block3_out", 
            "conv5_block3_3_conv", 
            "conv5_block3_add",
            "activation_49"  # Última ativação antes do pooling
        ]
        
        for camada_teste in opcoes_camadas:
            try:
                if hasattr(model.layers[0], 'get_layer'):
                    model.layers[0].get_layer(camada_teste)
                else:
                    model.get_layer(camada_teste)
                LAST_CONV_LAYER_NAME = camada_teste
                print(f"Usando camada: {LAST_CONV_LAYER_NAME}")
                break
            except ValueError:
                continue
        
        if LAST_CONV_LAYER_NAME is None:
            print("ERRO: Não foi possível encontrar uma camada convolucional válida.")
            print("Por favor, execute model.summary() para ver a estrutura completa.")
            exit(1)

    # 3. Configurações de pré-processamento
    IMG_SIZE = (224, 224)
    preprocess_input = keras.applications.resnet50.preprocess_input

    # 4. Define as imagens para análise
    caminho_base_teste = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", 
        "datasets", "francismon", "curated-colon-dataset-for-deep-learning",
        "versions", "1", "test"
    )

    imagens_para_analisar = {
        os.path.join(caminho_base_teste, "3_esophagitis", "test_esophagitis_ (23).jpg"): "gradcam_esophagitis.jpg",
        os.path.join(caminho_base_teste, "2_polyps", "test_polyps_ (23).jpg"): "gradcam_polipo.jpg",
        os.path.join(caminho_base_teste, "1_ulcerative_colitis", "ulcerative_colitis_ (23).jpg"): "gradcam_colite.jpg",
        os.path.join(caminho_base_teste, "0_normal", "test_normal_ (23).jpg"): "gradcam_normal.jpg",
    }
    
    # Se não encontrar os arquivos, tenta outros nomes
    if not any(os.path.exists(path) for path in imagens_para_analisar.keys()):
        print("Arquivos específicos não encontrados. Procurando automaticamente...")
        imagens_para_analisar = {}
        
        # Procura automaticamente por imagens em cada categoria
        categorias = ["0_normal", "1_ulcerative_colitis", "2_polyps"]
        for categoria in categorias:
            pasta = os.path.join(caminho_base_teste, categoria)
            if os.path.exists(pasta):
                arquivos = [f for f in os.listdir(pasta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if arquivos:
                    img_path = os.path.join(pasta, arquivos[0])  # Pega o primeiro arquivo
                    output_name = f"gradcam_{categoria}.jpg"
                    imagens_para_analisar[img_path] = output_name
                    print(f"Encontrado: {img_path}")
    
    # 5. Gera e salva as imagens Grad-CAM
    classes_nomes = {0: "Normal", 1: "Colite Ulcerativa", 2: "Pólipos", 3: "Outra"}
    
    for img_path, output_path in imagens_para_analisar.items():
        if not os.path.exists(img_path):
            print(f"[AVISO] Arquivo não encontrado, pulando: {img_path}")
            continue

        print(f"\n--- Analisando: {os.path.basename(img_path)} ---")
        
        # Prepara a imagem para o modelo
        img_array = get_img_array(img_path, size=IMG_SIZE)
        if img_array is None:
            continue
            
        img_preprocessed = preprocess_input(img_array.copy())

        # Faz a predição para mostrar a confiança
        preds = model.predict(img_preprocessed, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = preds[0][pred_class]
        
        print(f"Predição: {classes_nomes.get(pred_class, 'Desconhecida')} (confiança: {confidence:.3f})")

        # Gera o heatmap
        heatmap = make_gradcam_heatmap(img_preprocessed, model, LAST_CONV_LAYER_NAME)
        
        if heatmap is not None:
            # Salva a imagem com o heatmap sobreposto
            save_and_display_gradcam(img_path, heatmap, cam_path=output_path)
        else:
            print(f"Falha ao gerar heatmap para {img_path}")

    print("\n=== Análise Grad-CAM concluída! ===")
    print("Dicas para interpretação:")
    print("- Áreas vermelhas: Regiões mais importantes para a decisão")
    print("- Áreas azuis: Regiões menos relevantes")
    print("- Verifique se o modelo está focando nas regiões anatomicamente corretas")

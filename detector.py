import cv2
import numpy as np
import argparse
import time
from pathlib import Path

# ai_edge_litert é a biblioteca do Google para rodar modelos de IA leves (LiteRT = Lite Runtime).
# O TFLite (TensorFlow Lite) é uma versão compacta do TensorFlow feita para rodar em
# dispositivos com menos poder de processamento, como Raspberry Pi, celulares, etc.
# "Interpreter" é o objeto que carrega e executa o modelo dentro do Python.
from ai_edge_litert.interpreter import Interpreter


# Lista de cores em formato BGR (Blue, Green, Red — padrão do OpenCV, ao contrário do RGB comum).
# Cada classe detectada pelo modelo recebe uma cor diferente para ficar visualmente distinto.
CORES = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def carregar_labels(caminho):
    """
    Carrega os labels (nomes das classes) do arquivo .txt.
    
    O arquivo de labels é uma lista simples onde cada linha é o nome de um objeto
    que o modelo sabe detectar — por exemplo: 'person', 'car', 'dog', etc.
    
    ⚠️ BUG POTENCIAL: Se o arquivo estiver vazio, `labels[0]` vai lançar um
    IndexError. Seria seguro adicionar um `if labels:` antes da verificação.
    """
    with open(caminho, "r") as f:
        labels = [linha.strip() for linha in f.readlines()]

    # Alguns arquivos de label do TFLite começam com '???' como placeholder vazio.
    # Esse item é removido para não interferir nos índices reais das classes.
    if labels[0] == "???":
        labels.pop(0)

    return labels


def carregar_modelo(caminho_modelo):
    """
    Carrega o arquivo .tflite e prepara o modelo para rodar inferências.
    
    Um modelo TFLite é basicamente uma rede neural treinada e compactada num
    único arquivo binário. O Interpreter lê esse arquivo e cria uma estrutura
    interna que sabe executar as operações do modelo.
    
    "Alocar tensores" significa reservar a memória necessária para os dados
    de entrada e saída — isso precisa ser feito uma vez antes de usar o modelo.
    
    input_details  → descreve o que o modelo espera receber (tamanho da imagem, tipo de dado)
    output_details → descreve o que o modelo vai retornar (caixas, classes, scores, total)
    """
    interpreter = Interpreter(model_path=str(caminho_modelo))
    interpreter.allocate_tensors()  # Reserva memória para os tensores

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # O modelo TFLite de detecção espera imagens num tamanho fixo (ex: 300x300).
    # Esse tamanho é lido diretamente dos metadados do modelo.
    altura = input_details[0]["shape"][1]
    largura = input_details[0]["shape"][2]

    return interpreter, input_details, output_details, altura, largura


def preprocessar_frame(frame, largura, altura):
    """
    Prepara um frame da câmera para ser enviado ao modelo.
    
    Três passos necessários:
    1. Converter BGR → RGB: OpenCV captura em BGR, mas o modelo foi treinado com RGB.
       Sem isso, as cores ficam invertidas e a detecção piora bastante.
    2. Redimensionar: o modelo só aceita imagens no tamanho exato que foi treinado.
    3. Adicionar dimensão de batch: modelos TFLite esperam um array 4D no formato
       [batch, altura, largura, canais]. O `expand_dims` transforma (H, W, 3) em (1, H, W, 3).
    
    ⚠️ BUG POTENCIAL: O `.astype(np.uint8)` assume que o modelo aceita inteiros de 8 bits
    (0–255). Modelos não-quantizados esperam float32 (valores entre 0.0 e 1.0).
    Se o modelo for float, isso vai gerar detecções incorretas ou erro silencioso.
    O correto seria verificar `input_details[0]['dtype']` antes de converter.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_redim = cv2.resize(img_rgb, (largura, altura))
    return np.expand_dims(img_redim, axis=0).astype(np.uint8)


def detectar(interpreter, input_details, output_details, img):
    """
    Executa a inferência (o "pensamento" do modelo) sobre a imagem.
    
    set_tensor → "coloca" a imagem dentro do modelo na entrada correta.
    invoke()   → executa o modelo. É aqui que a rede neural roda de fato.
    get_tensor → lê os resultados de cada saída depois da inferência.
    
    O modelo de detecção retorna 4 tensores (saídas):
      [0] caixas    → coordenadas normalizadas [ymin, xmin, ymax, xmax] de cada detecção
      [1] classes   → índice da classe detectada (ex: 0 = person, 2 = car)
      [2] scores    → confiança de cada detecção (0.0 a 1.0)
      [3] total     → número de detecções encontradas
    
    ⚠️ BUG POTENCIAL: A ordem dos tensores de saída (0, 1, 2, 3) não é garantida e
    varia dependendo de como o modelo foi exportado. Modelos diferentes podem ter
    caixas no índice 1 e classes no índice 0, por exemplo. O correto seria verificar
    os nomes dos tensores via `output_details[i]['name']` para mapear corretamente.
    
    ⚠️ ATENÇÃO: A variável `total` retornada aqui nunca é usada na função `main()`.
    Ela é retornada mas ignorada — o loop em `desenhar_deteccoes` usa `len(scores)`
    em vez disso. Não é um erro grave (o filtro por limiar cobre), mas é desperdício.
    """
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    caixas = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    total = int(interpreter.get_tensor(output_details[3]["index"])[0])

    return caixas, classes, scores, total


def desenhar_deteccoes(frame, caixas, classes, scores, labels, limiar=0.5):
    """
    Desenha as caixas delimitadoras (bounding boxes) e os labels no frame.
    
    Para cada objeto detectado com confiança acima do limiar:
    - Converte as coordenadas normalizadas (0.0 a 1.0) para pixels reais
    - Desenha o retângulo colorido ao redor do objeto
    - Escreve o nome da classe e a porcentagem de confiança
    - Conta quantos objetos de cada classe foram detectados
    
    As coordenadas são normalizadas porque o modelo foi treinado com imagens
    redimensionadas — ao normalizar, as coordenadas funcionam em qualquer resolução.
    """
    h, w = frame.shape[:2]
    contagem = {}

    for i in range(len(scores)):
        # Ignora detecções com confiança abaixo do limiar definido pelo usuário
        if scores[i] < limiar:
            continue

        idx_classe = int(classes[i])

        # Proteção: se o índice da classe for maior que o número de labels,
        # usa um nome genérico em vez de travar com IndexError
        nome = labels[idx_classe] if idx_classe < len(labels) else f"classe_{idx_classe}"
        cor = CORES[idx_classe % len(CORES)]  # % garante que não ultrapassa a lista de cores
        confianca = int(scores[i] * 100)

        # Converte coordenadas normalizadas para coordenadas em pixels
        # O modelo retorna [ymin, xmin, ymax, xmax] — note a ordem Y antes de X
        ymin, xmin, ymax, xmax = caixas[i]
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        # Desenha o retângulo ao redor do objeto detectado (espessura 2 pixels)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        # Mede o tamanho do texto para criar o fundo colorido proporcional
        texto = f"{nome}: {confianca}%"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        # Retângulo preenchido (-1) atrás do texto para melhorar a legibilidade
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), cor, -1)

        # Escreve o texto em branco sobre o fundo colorido
        cv2.putText(frame, texto, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Incrementa o contador da classe detectada
        contagem[nome] = contagem.get(nome, 0) + 1

    return frame, contagem


def desenhar_painel(frame, contagem, fps):
    """
    Desenha um painel semitransparente com estatísticas no canto superior esquerdo.
    
    Usa `addWeighted` para criar o efeito de transparência:
    - Copia o frame original em `overlay`
    - Desenha o retângulo sólido na cópia
    - Mistura overlay (70%) com o frame original (30%)
    Resultado: fundo escuro semitransparente, sem apagar o que está atrás.
    
    ⚠️ BUG POTENCIAL: Se `contagem` tiver muitos itens, o painel pode ultrapassar
    a borda inferior do frame. Não há verificação de `painel_altura <= h`.
    Isso causaria que o texto fosse cortado ou aparecesse fora da tela.
    """
    h, w = frame.shape[:2]
    painel_largura = 220
    painel_altura = 30 + len(contagem) * 24 + 30
    painel_altura = max(painel_altura, 80)  # Garante altura mínima de 80px

    # Técnica de overlay semitransparente com OpenCV
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (painel_largura, painel_altura), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)  # 70% overlay + 30% original

    # Exibe o FPS calculado na função main()
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

    # Total de objetos detectados neste frame (soma de todos os contadores)
    total = sum(contagem.values())
    cv2.putText(frame, f"Objetos: {total}", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Lista cada classe detectada e sua quantidade, ordenada da mais para menos frequente
    y = 80
    for nome, qtd in sorted(contagem.items(), key=lambda x: -x[1]):
        texto = f"  {nome}: {qtd}"
        cv2.putText(frame, texto, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        y += 22

    return frame


def main():
    """
    Função principal: configura os argumentos, carrega o modelo e roda o loop de detecção.
    
    O fluxo geral é:
    1. Lê os argumentos da linha de comando
    2. Carrega o modelo e os labels
    3. Abre a câmera ou o arquivo de vídeo
    4. A cada frame: pré-processa → detecta → desenha → exibe
    5. Calcula FPS a cada 10 frames para maior estabilidade
    """
    parser = argparse.ArgumentParser(description="Detector de objetos em tempo real")
    parser.add_argument("--modelo", default="models/detect.tflite", help="Caminho do modelo .tflite")
    parser.add_argument("--labels", default="models/labelmap.txt", help="Caminho do arquivo de labels")
    parser.add_argument("--limiar", type=float, default=0.5, help="Confiança mínima (0.0 a 1.0)")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera USB")
    parser.add_argument("--video", type=str, default=None, help="Caminho de um vídeo gravado")
    parser.add_argument("--largura", type=int, default=640, help="Largura do frame de exibição")
    args = parser.parse_args()

    print("Carregando modelo...")
    labels = carregar_labels(args.labels)
    interpreter, input_details, output_details, altura_modelo, largura_modelo = carregar_modelo(args.modelo)
    print(f"Modelo carregado. Entrada: {largura_modelo}x{altura_modelo}")

    # Se --video foi passado, usa o arquivo de vídeo; senão, usa a câmera pelo índice
    fonte = args.video if args.video else args.camera
    cap = cv2.VideoCapture(fonte)

    if not cap.isOpened():
        print(f"Erro: não foi possível abrir a fonte de vídeo: {fonte}")
        return

    # Define a largura de captura (pode ser ignorada pela câmera dependendo do driver)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.largura)

    print("Iniciando detecção... Pressione 'q' para sair.")

    # Variáveis para calcular FPS de forma estável (média a cada 10 frames)
    fps = 0
    tempo_anterior = time.time()
    contador_frames = 0

    while True:
        ret, frame = cap.read()

        # `ret` é False quando a câmera desconecta ou o vídeo termina
        if not ret:
            print("Fim do vídeo ou erro na câmera.")
            break

        # Pipeline completo por frame
        img = preprocessar_frame(frame, largura_modelo, altura_modelo)
        
        # `total` aqui é retornado mas nunca usado — ver comentário em `detectar()`
        caixas, classes, scores, total = detectar(interpreter, input_details, output_details, img)

        frame, contagem = desenhar_deteccoes(frame, caixas, classes, scores, labels, args.limiar)

        # Atualiza o FPS a cada 10 frames para evitar oscilação numérica
        contador_frames += 1
        if contador_frames >= 10:
            tempo_atual = time.time()
            fps = contador_frames / (tempo_atual - tempo_anterior)
            tempo_anterior = tempo_atual
            contador_frames = 0

        frame = desenhar_painel(frame, contagem, fps)

        cv2.imshow("Detector de Objetos", frame)

        # waitKey(1) aguarda 1ms por input do teclado — necessário para o OpenCV atualizar a janela
        # O `& 0xFF` isola o byte menos significativo para compatibilidade com sistemas 64-bit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Encerrando...")
            break

    # Libera a câmera e fecha todas as janelas do OpenCV
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Esse bloco garante que main() só roda quando o script é executado diretamente,
    # não quando é importado como módulo por outro arquivo Python
    main()
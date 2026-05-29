import cv2
import numpy as np
import argparse
import time
import threading
from ai_edge_litert.interpreter import Interpreter


CORES = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


class CameraThread:
    """
    Captura frames em uma thread separada para não bloquear a inferência.
    
    Sem threading: captura → inferência → exibe → captura → ...
    Com threading: captura roda continuamente em paralelo com inferência.
    
    O frame mais recente fica disponível em `self.frame` a qualquer momento.
    O lock (mutex) evita que a thread de leitura e a de inferência acessem
    o mesmo frame simultaneamente, o que causaria corrupção de dados.
    """
    def __init__(self, fonte, largura):
        self.cap = cv2.VideoCapture(fonte)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(largura * 0.75))  # 4:3 aspect ratio
        
        # Reduz o buffer interno para sempre pegar o frame mais recente
        # Sem isso, o OpenCV acumula frames antigos e gera delay visível
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True

        # Inicia a thread de captura em background (daemon=True faz ela morrer com o programa)
        self.thread = threading.Thread(target=self._capturar, daemon=True)
        self.thread.start()

    def _capturar(self):
        """Loop de captura que roda em background continuamente."""
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def ler(self):
        """Retorna o frame mais recente capturado pela thread."""
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def parar(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    def aberta(self):
        return self.cap.isOpened()


def carregar_labels(caminho):
    with open(caminho, "r") as f:
        labels = [linha.strip() for linha in f.readlines()]
    if labels and labels[0] == "???":
        labels.pop(0)
    return labels


def carregar_modelo(caminho_modelo):
    interpreter = Interpreter(model_path=str(caminho_modelo))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    altura = input_details[0]["shape"][1]
    largura = input_details[0]["shape"][2]
    return interpreter, input_details, output_details, altura, largura


def preprocessar_frame(frame, largura, altura):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_redim = cv2.resize(img_rgb, (largura, altura))
    return np.expand_dims(img_redim, axis=0).astype(np.uint8)


def detectar(interpreter, input_details, output_details, img):
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    caixas = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    return caixas, classes, scores


def desenhar_deteccoes(frame, caixas, classes, scores, labels, limiar=0.5):
    h, w = frame.shape[:2]
    contagem = {}

    for i in range(len(scores)):
        if scores[i] < limiar:
            continue

        idx_classe = int(classes[i])
        nome = labels[idx_classe] if idx_classe < len(labels) else f"classe_{idx_classe}"
        cor = CORES[idx_classe % len(CORES)]
        confianca = int(scores[i] * 100)

        ymin, xmin, ymax, xmax = caixas[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        texto = f"{nome}: {confianca}%"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), cor, -1)
        cv2.putText(frame, texto, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        contagem[nome] = contagem.get(nome, 0) + 1

    return frame, contagem


def desenhar_painel(frame, contagem, fps):
    h, w = frame.shape[:2]
    painel_largura = 220
    painel_altura = max(30 + len(contagem) * 24 + 30, 80)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (painel_largura, painel_altura), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

    total = sum(contagem.values())
    cv2.putText(frame, f"Objetos: {total}", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    y = 80
    for nome, qtd in sorted(contagem.items(), key=lambda x: -x[1]):
        cv2.putText(frame, f"  {nome}: {qtd}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        y += 22

    return frame


def main():
    parser = argparse.ArgumentParser(description="Detector de objetos em tempo real")
    parser.add_argument("--modelo", default="models/detect.tflite")
    parser.add_argument("--labels", default="models/labelmap.txt")
    parser.add_argument("--limiar", type=float, default=0.5)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--largura", type=int, default=640)
    parser.add_argument("--skip", type=int, default=2,
                        help="Roda inferência a cada N frames (padrão: 2). "
                             "Valores maiores = mais FPS, menos precisão temporal.")
    args = parser.parse_args()

    print("Carregando modelo...")
    labels = carregar_labels(args.labels)
    interpreter, input_details, output_details, altura_modelo, largura_modelo = carregar_modelo(args.modelo)
    print(f"Modelo carregado. Entrada: {largura_modelo}x{altura_modelo}")

    fonte = args.video if args.video else args.camera

    # Usa CameraThread para captura em background (elimina delay de captura)
    # Para vídeos gravados, o threading tem menos impacto mas ainda ajuda
    camera = CameraThread(fonte, args.largura)

    if not camera.aberta():
        print(f"Erro: não foi possível abrir a fonte de vídeo: {fonte}")
        return

    print(f"Iniciando detecção (skip={args.skip})... Pressione 'q' para sair.")

    fps = 0
    tempo_anterior = time.time()
    contador_frames = 0

    # Guarda o último resultado de inferência para reusar nos frames pulados
    ultimo_resultado = (np.zeros((10, 4)), np.zeros(10), np.zeros(10))

    while True:
        ret, frame = camera.ler()
        if not ret or frame is None:
            print("Fim do vídeo ou erro na câmera.")
            break

        contador_frames += 1

        # Roda inferência apenas a cada `skip` frames
        # Nos frames intermediários, reutiliza o último resultado — sem delay de modelo
        if contador_frames % args.skip == 0:
            img = preprocessar_frame(frame, largura_modelo, altura_modelo)
            caixas, classes, scores = detectar(interpreter, input_details, output_details, img)
            ultimo_resultado = (caixas, classes, scores)
        else:
            caixas, classes, scores = ultimo_resultado

        frame, contagem = desenhar_deteccoes(frame, caixas, classes, scores, labels, args.limiar)

        # Calcula FPS a cada 30 frames para maior estabilidade
        if contador_frames % 30 == 0:
            tempo_atual = time.time()
            fps = 30 / (tempo_atual - tempo_anterior)
            tempo_anterior = tempo_atual

        frame = desenhar_painel(frame, contagem, fps)

        cv2.imshow("Detector de Objetos", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Encerrando...")
            break

    camera.parar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
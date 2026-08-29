#!/usr/bin/env python3
"""Sobe tudo de uma vez: backend, frontend e o mapa do projeto.

Uso:
    python iniciar.py                 # os três, e abre o app no navegador
    python iniciar.py --mapa          # abre o mapa do projeto em vez do app
    python iniciar.py --sem-mapa      # só backend e frontend

    http://localhost:8001   backend FastAPI (sinal, filtro, eventos)
    http://localhost:5500   frontend do app EEG
    http://localhost:8002   mapa do mestrado

Ctrl+C encerra todos.

POR QUE OS TRÊS JUNTOS
----------------------
O mapa lê o repositório e o vault, não o backend, então tecnicamente ele roda
sozinho. Sobe junto assim mesmo porque a pergunta que ele responde ("em que pé
está isto?") é a primeira que se faz ao abrir o projeto, e um serviço que
exige um comando extra é um serviço que ninguém abre.

O QUE ESTE SCRIPT APRENDEU A NÃO FAZER
--------------------------------------
Ele já derrubou os três serviços dizendo apenas "backend caiu". A causa real
era a porta 8001 estar ocupada por um backend de uma sessão anterior: o
uvicorn não conseguia escutar, morria em um segundo, e este script matava o
frontend e o mapa junto — os dois funcionando perfeitamente. Quem visse isso
não teria como saber que o problema era porta ocupada.

Agora ele olha cada porta ANTES de subir, diz o que encontrou, e reaproveita
o que já está de pé em vez de brigar por ele.
"""
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Uma linha, e ela conserta todo print deste arquivo.
#
# Quando este script roda num terminal, o Python usa buffer de linha e tudo
# aparece na hora. Quando ele roda com a saída redirecionada — um arquivo de
# log, um `| tee`, uma janela de tarefa em segundo plano — o Python troca
# para buffer de bloco de alguns KB, e as poucas linhas daqui ficam presas
# nele até o processo terminar. O efeito prático é o pior possível: o
# lançador parece não fazer nada, e a mensagem que explica o que deu errado
# só aparece depois que já não adianta.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass   # stdout exótico (pytest, embutido): segue com o padrão

RAIZ = Path(__file__).resolve().parent
PASTA_BACKEND = RAIZ / "backend"

PORTA_BACKEND = 8001
PORTA_FRONTEND = 5500
PORTA_MAPA = 8002

URL_APP = f"http://localhost:{PORTA_FRONTEND}/eeg-cerebro-3d.html"
URL_MAPA = f"http://localhost:{PORTA_MAPA}/"


def porta_ocupada(porta):
    """True se já existe alguém escutando nesta porta.

    Tentar conectar é mais confiável que tentar escutar: no Windows, o
    SO_REUSEADDR deixa dois processos ligarem na mesma porta em algumas
    situações, e o segundo simplesmente nunca recebe requisição."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", porta)) == 0


def esperar_porta(porta, segundos):
    """Espera a porta começar a atender, e devolve se conseguiu.

    Existe porque "o processo não morreu" não é o mesmo que "o serviço está
    de pé": o uvicorn leva um tempo carregando o CSV de 267 MB e construindo
    o modelo de fonte, e abrir o navegador antes disso mostra uma tela vazia
    que parece defeito."""
    limite = time.time() + segundos
    while time.time() < limite:
        if porta_ocupada(porta):
            return True
        time.sleep(0.3)
    return False


def subir(rotulo, comando, cwd, porta):
    """Sobe um serviço, ou reaproveita o que já está na porta.

    Devolve o processo, ou None quando a porta já estava ocupada. None não é
    falha: é "isto já está de pé e não é meu para gerenciar", e o laço
    principal não tenta vigiar o que não subiu."""
    if porta_ocupada(porta):
        print(f"[iniciar] {rotulo}: porta {porta} JÁ está em uso — reaproveitando o que está lá.")
        print(f"[iniciar]   (se for de outra sessão e estiver velho, encerre-a e rode de novo)")
        return None
    print(f"[iniciar] {rotulo}")
    return subprocess.Popen(comando, cwd=cwd)


def main():
    com_mapa = "--sem-mapa" not in sys.argv
    abrir_mapa = "--mapa" in sys.argv

    servicos = []   # (nome, processo|None, porta, essencial)

    servicos.append(("backend", subir(
        f"backend  → http://localhost:{PORTA_BACKEND}",
        [sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORTA_BACKEND)],
        PASTA_BACKEND, PORTA_BACKEND), PORTA_BACKEND, True))

    servicos.append(("frontend", subir(
        f"frontend → {URL_APP}",
        # --bind 127.0.0.1 NÃO é detalhe. Sem ele o http.server escuta em
        # 0.0.0.0 e em [::], e serve a RAIZ DO PROJETO inteira: uma auditoria
        # baixou os 254,9 MB do adhdata.csv pelo IP da Wi-Fi, sem
        # autenticação, com o rótulo ADHD/Control colado ao identificador em
        # toda linha. São 121 crianças de 7 a 12 anos. A mesma porta respondia
        # pelo IP do Tailscale, ou seja, de fora da rede local.
        #
        # O uvicorn e o painel de progresso já escutavam só em 127.0.0.1; o
        # frontend era o único aberto, e era o que servia o dado bruto.
        [sys.executable, "-m", "http.server", str(PORTA_FRONTEND), "--bind", "127.0.0.1"],
        RAIZ, PORTA_FRONTEND), PORTA_FRONTEND, True))

    if com_mapa:
        servicos.append(("mapa", subir(
            f"mapa     → {URL_MAPA}",
            [sys.executable, "scripts/painel_progresso.py", "--porta", str(PORTA_MAPA)],
            PASTA_BACKEND, PORTA_MAPA), PORTA_MAPA, False))

    # O backend carrega 267 MB de CSV e monta o modelo de fonte: na primeira
    # execução da máquina isso passa de um minuto. Esperar a porta atender é
    # o que evita abrir o navegador numa tela que ainda não tem dado.
    print("[iniciar] esperando os serviços atenderem…")
    faltou = []
    for nome, _proc, porta, essencial in servicos:
        if not esperar_porta(porta, 180 if nome == "backend" else 20):
            faltou.append((nome, porta, essencial))

    if faltou:
        print()
        for nome, porta, essencial in faltou:
            print(f"[iniciar] {nome} NÃO subiu na porta {porta}"
                  f"{' (essencial)' if essencial else ' (acessório)'}")
        if any(e for _n, _p, e in faltou):
            print("[iniciar] Sem os serviços essenciais não vale abrir o navegador.")
            print("[iniciar] Cheque a saída acima: erro de import, dependência faltando"
                  " ou porta tomada por outro programa.")
            encerrar(servicos)
            return

    alvo = URL_MAPA if (abrir_mapa and com_mapa) else URL_APP
    print(f"[iniciar] abrindo {alvo}")
    webbrowser.open(alvo)

    print("[iniciar] tudo rodando. Ctrl+C para encerrar.")
    try:
        while True:
            time.sleep(1)
            for nome, proc, porta, essencial in list(servicos):
                if proc is None or proc.poll() is None:
                    continue
                # Um serviço morreu. Só derruba o resto se ele for essencial:
                # o mapa caindo não é motivo para fechar o app do usuário.
                if essencial:
                    print(f"[iniciar] {nome} caiu (porta {porta}). Encerrando o resto.")
                    raise KeyboardInterrupt
                print(f"[iniciar] o {nome} caiu. O app segue rodando.")
                servicos = [s for s in servicos if s[0] != nome]
    except KeyboardInterrupt:
        print("\n[iniciar] encerrando...")
    finally:
        encerrar(servicos)


def encerrar(servicos):
    """Encerra só o que ESTE processo subiu.

    Serviço reaproveitado tem proc None e não é tocado: matar o backend de
    outra sessão seria efeito colateral que ninguém pediu."""
    meus = [p for _n, p, _pt, _e in servicos if p is not None]
    for proc in meus:
        if proc.poll() is None:
            proc.terminate()
    for proc in meus:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()

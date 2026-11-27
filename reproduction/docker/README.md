# Reprodução em Docker (CPU)

Este container executa a avaliação com predições já calculadas. Ele não treina
modelos, não usa CUDA e não precisa acessar a GPU do servidor.

## 1. Levar os arquivos ao servidor

O clone remoto precisa conter esta pasta `reproduction/docker`. Faça o push do
commit local e `git pull` no servidor, ou copie a pasta com `rsync`/`scp` se não
quiser publicar o commit ainda.

Entre na raiz do repositório remoto antes dos próximos comandos.

## 2. Construir a imagem

```bash
sudo env LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" \
  docker compose -f reproduction/docker/compose.yaml build reproduction
```

A imagem usa Python 3.10 e PyTorch 2.5 CPU. O contexto de construção contém
somente os arquivos pequenos de `reproduction/docker`; dados e predições são
montados em tempo de execução.

## 3. Iniciar em segundo plano

```bash
sudo env LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" RUNS=10 CORES=8 \
  docker compose -f reproduction/docker/compose.yaml up -d reproduction
```

Você pode encerrar a sessão SSH depois que o serviço estiver em execução.

## 4. Acompanhar

```bash
sudo docker compose -f reproduction/docker/compose.yaml ps -a
sudo docker compose -f reproduction/docker/compose.yaml logs -f reproduction
```

Use `Ctrl+C` para sair da visualização dos logs; isso não interrompe o
container. O log persistente também fica em
`reproduction/logs/docker-reproduction-runs-10.log`.

## 5. Resultado

Ao terminar com sucesso, o CSV estará em:

```text
reproduction/results/main/results_kuairand_hybrid_lightgcl_harm_10_0.0_0.3_100_False_all.csv
```

O script não sobrescreve resultado ou log existentes. Se precisar repetir a
execução, preserve ou renomeie os arquivos anteriores primeiro.
Execuções simultâneas com o mesmo número de runs são bloqueadas. Os splits são
gravados atomicamente e verificados por hashes antes da avaliação; uma geração
interrompida é refeita na próxima tentativa.

## Teste curto opcional

Antes das dez repetições, é possível validar uma repetição em outra pasta:

```bash
mkdir -p reproduction/results/container-smoke reproduction/logs/container-smoke
sudo env LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" RUNS=1 CORES=8 \
  OUTPUT_DIR=/workspace/reproduction/results/container-smoke \
  LOG_DIR=/workspace/reproduction/logs/container-smoke \
  docker compose -f reproduction/docker/compose.yaml run --rm reproduction
```

## Interromper

```bash
sudo docker compose -f reproduction/docker/compose.yaml stop reproduction
```

Uma interrupção não cria um CSV no caminho final; o resultado só é movido
depois que o avaliador termina com sucesso.

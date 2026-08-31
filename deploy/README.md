# Implantação

Arquivos de configuração para executar o serviço de inferência fora da máquina de desenvolvimento. O escopo é acadêmico: **nada aqui foi provisionado** e nenhum recurso pago foi criado.

O que é implantado é a **execução de um modelo já congelado**. Não há treino, otimização, nem consulta ao holdout em nenhum destes caminhos.

## Container

```bash
docker build -t fiap-techchallenge-fase2-inferencia:0.8.0 .
docker run --rm fiap-techchallenge-fase2-inferencia:0.8.0
```

A imagem fixa uma thread de BLAS por réplica. Sem isso a álgebra linear paraleliza dentro do processo, satura as CPUs com uma réplica e anula o efeito de escalar — o achado documentado em [`../docs/escalabilidade_e_monitoramento.md`](../docs/escalabilidade_e_monitoramento.md).

O `HEALTHCHECK` confere o SHA-256 do pipeline congelado contra o manifesto assinado. Hash divergente derruba o container em vez de servir um modelo desconhecido.

## Orquestração local

```bash
docker compose up --build
```

`deploy.replicas` reproduz localmente o recurso que o autoscaling controla na nuvem: número de réplicas do processo.

## Nuvem

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars   # preencher com valores da conta
terraform init
terraform plan
```

| Recurso | Papel |
|---|---|
| `aws_ecs_cluster` | Cluster Fargate com Container Insights habilitado |
| `aws_ecs_task_definition` | Tarefa com BLAS fixado e healthcheck de integridade do modelo |
| `aws_ecs_service` | Serviço em subnets privadas, sem IP público e sem balanceador |
| `aws_appautoscaling_target` | Piso e teto de tarefas, espelhando `min_workers` e `max_workers` |
| `aws_appautoscaling_policy` | Target tracking de CPU, com cooldowns assimétricos |
| `aws_cloudwatch_log_group` | Recebe os eventos de desempenho |

O piso e o teto de réplicas espelham deliberadamente a política em `src/tech_challenge_fase2/serving/autoscaling.py`: o recurso escalado na nuvem é o mesmo que o benchmark local mede.

Os cooldowns assimétricos — 60 s para subir, 300 s para descer — são o equivalente na nuvem da histerese da política local. Descer depressa demais devolve a fila ao gargalo que acabou de ser aliviado.

`terraform fmt` e `terraform validate` rodam no CI a cada push, então a configuração é verificada mesmo sem ser aplicada.

## O que não está aqui

- Nenhum `terraform apply` foi executado, então não há evidência de comportamento em nuvem real.
- Não há endpoint público, autenticação de API nem balanceador: o serviço processa lotes, não requisições de terceiros.
- Nenhum dado de paciente é copiado para a imagem além da base pública já versionada no repositório.

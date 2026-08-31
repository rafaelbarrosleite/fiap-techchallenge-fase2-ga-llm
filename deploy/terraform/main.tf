# Execucao do modelo congelado em ECS Fargate, com autoscaling de tarefas.
#
# O recurso escalado e o numero de tarefas do servico, o mesmo que a politica
# local controla como workers. Manter os dois alinhados e proposital: o
# benchmark local mede o comportamento que esta configuracao provisiona.
#
# Escopo academico. Este modulo provisiona apenas a execucao de um modelo ja
# treinado e congelado. Ele nao treina, nao versiona dados de paciente e nao
# expoe endpoint publico: as tarefas correm em subnets privadas, sem IP
# publico e sem balanceador.

provider "aws" {
  region = var.regiao
}

resource "aws_ecs_cluster" "this" {
  name = var.nome

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "this" {
  name              = "/ecs/${var.nome}"
  retention_in_days = var.retencao_de_logs_em_dias
}

resource "aws_ecs_task_definition" "this" {
  family                   = var.nome
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = var.execution_role_arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = "inferencia"
      image     = var.imagem
      essential = true

      # Uma thread de BLAS por replica. Sem isso a algebra linear paraleliza
      # dentro do processo, satura as CPUs da tarefa e anula o efeito de
      # escalar tarefas -- o mesmo achado registrado no benchmark local.
      environment = [
        { name = "OMP_NUM_THREADS", value = "1" },
        { name = "OPENBLAS_NUM_THREADS", value = "1" },
        { name = "MKL_NUM_THREADS", value = "1" }
      ]

      # Falha cedo se o pipeline congelado nao conferir com o manifesto
      # assinado, em vez de servir um modelo desconhecido.
      healthCheck = {
        command = [
          "CMD-SHELL",
          "uv run python -c 'from tech_challenge_fase2.serving import resolve_frozen_model; resolve_frozen_model()'"
        ]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 15
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.this.name
          "awslogs-region"        = var.regiao
          "awslogs-stream-prefix" = "inferencia"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "this" {
  name            = var.nome
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.replicas_minimas
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = var.security_group_ids
    assign_public_ip = false
  }

  # O autoscaling passa a ser dono de desired_count depois do provisionamento;
  # sem isso, cada apply devolveria o servico ao piso.
  lifecycle {
    ignore_changes = [desired_count]
  }
}

resource "aws_appautoscaling_target" "this" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.this.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.replicas_minimas
  max_capacity       = var.replicas_maximas
}

# Cooldowns assimetricos sao o equivalente na nuvem da histerese da politica
# local: descer depressa demais devolve a fila ao gargalo que acabou de ser
# aliviado, entao a descida espera cinco vezes mais que a subida.
resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.nome}-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.this.service_namespace
  resource_id        = aws_appautoscaling_target.this.resource_id
  scalable_dimension = aws_appautoscaling_target.this.scalable_dimension

  target_tracking_scaling_policy_configuration {
    target_value       = var.cpu_alvo_percentual
    scale_in_cooldown  = 300
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

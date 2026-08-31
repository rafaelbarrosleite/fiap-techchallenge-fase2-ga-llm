# Infraestrutura como codigo para o servico de inferencia com autoscaling.
#
# O recurso escalado e o numero de tarefas do servico, o mesmo que a politica
# local controla como workers. As metricas de alvo espelham a politica de
# src/tech_challenge_fase2/serving/autoscaling.py: subir sob pressao, descer na
# ociosidade e respeitar um teto explicito.
#
# Escopo academico. Este modulo provisiona a execucao do modelo ja congelado;
# ele nao treina, nao versiona dados de paciente e nao expoe endpoint publico.

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.regiao
}

variable "regiao" {
  description = "Regiao AWS onde o servico e provisionado."
  type        = string
  default     = "us-east-1"
}

variable "nome" {
  description = "Prefixo de nome dos recursos."
  type        = string
  default     = "fiap-fase2-inferencia"
}

variable "imagem" {
  description = "Imagem do container publicada em um registro acessivel ao cluster."
  type        = string
}

variable "replicas_minimas" {
  description = "Piso de tarefas; espelha min_workers da politica local."
  type        = number
  default     = 1
}

variable "replicas_maximas" {
  description = "Teto de tarefas; espelha max_workers da politica local."
  type        = number
  default     = 4
}

variable "cpu_alvo_percentual" {
  description = "Utilizacao de CPU perseguida pelo autoscaling."
  type        = number
  default     = 60
}

resource "aws_ecs_cluster" "this" {
  name = var.nome
}

resource "aws_cloudwatch_log_group" "this" {
  name              = "/ecs/${var.nome}"
  retention_in_days = 30
}

resource "aws_ecs_task_definition" "this" {
  family                   = var.nome
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = var.execution_role_arn

  container_definitions = jsonencode([
    {
      name      = "inferencia"
      image     = var.imagem
      essential = true
      # Uma thread de BLAS por replica: o paralelismo vem do autoscaling.
      environment = [
        { name = "OMP_NUM_THREADS", value = "1" },
        { name = "OPENBLAS_NUM_THREADS", value = "1" },
        { name = "MKL_NUM_THREADS", value = "1" }
      ]
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

variable "execution_role_arn" {
  description = "Role de execucao das tarefas ECS."
  type        = string
}

variable "subnet_ids" {
  description = "Subnets privadas onde as tarefas correm."
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security groups aplicados as tarefas."
  type        = list(string)
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
}

resource "aws_appautoscaling_target" "this" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.this.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.replicas_minimas
  max_capacity       = var.replicas_maximas
}

# Subida sob pressao e descida na ociosidade, com cooldowns distintos: descer
# depressa demais devolve a fila ao mesmo gargalo que acabou de ser aliviado.
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

output "cluster" {
  description = "Cluster ECS provisionado."
  value       = aws_ecs_cluster.this.name
}

output "faixa_de_replicas" {
  description = "Piso e teto de tarefas sob autoscaling."
  value       = "${var.replicas_minimas}-${var.replicas_maximas}"
}

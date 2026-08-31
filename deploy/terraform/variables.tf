# Entradas do modulo. Os limites de replicas espelham deliberadamente
# min_workers e max_workers de src/tech_challenge_fase2/serving/autoscaling.py:
# o recurso escalado na nuvem e o mesmo que a politica local controla.

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

variable "replicas_minimas" {
  description = "Piso de tarefas; espelha min_workers da politica local."
  type        = number
  default     = 1

  validation {
    condition     = var.replicas_minimas >= 1
    error_message = "O piso precisa ser de pelo menos uma tarefa, como min_workers."
  }
}

variable "replicas_maximas" {
  description = "Teto de tarefas; espelha max_workers da politica local."
  type        = number
  default     = 4

  validation {
    condition     = var.replicas_maximas >= 1
    error_message = "O teto precisa ser de pelo menos uma tarefa."
  }
}

variable "cpu_alvo_percentual" {
  description = "Utilizacao de CPU perseguida pelo autoscaling."
  type        = number
  default     = 60

  validation {
    condition     = var.cpu_alvo_percentual > 0 && var.cpu_alvo_percentual <= 100
    error_message = "O alvo de CPU precisa estar entre 1 e 100 por cento."
  }
}

variable "retencao_de_logs_em_dias" {
  description = "Retencao do grupo de logs de desempenho."
  type        = number
  default     = 30
}

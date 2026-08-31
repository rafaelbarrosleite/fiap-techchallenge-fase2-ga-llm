output "cluster" {
  description = "Cluster ECS provisionado."
  value       = aws_ecs_cluster.this.name
}

output "servico" {
  description = "Servico ECS que executa o modelo congelado."
  value       = aws_ecs_service.this.name
}

output "faixa_de_replicas" {
  description = "Piso e teto de tarefas sob autoscaling."
  value       = "${var.replicas_minimas}-${var.replicas_maximas}"
}

output "grupo_de_logs" {
  description = "Grupo do CloudWatch que recebe os eventos de desempenho."
  value       = aws_cloudwatch_log_group.this.name
}

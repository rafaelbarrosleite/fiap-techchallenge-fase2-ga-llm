"""Trilha historica preservada da integracao com o provider real.

Estes modulos nao fazem parte do fluxo oficial da entrega. Eles registram, na
ordem em que aconteceram, o diagnostico de parametros incompativeis da
Responses API, a correcao do parsing, a criacao do contrato V2 e a unica
avaliacao real executada. O conteudo permanece executavel e testado porque a
evidencia de falha faz parte do relatorio: o enunciado pede desafios
enfrentados e solucoes implementadas, e apagar as tentativas apagaria a
resposta.

O subpacote existe para separar essa trilha do codigo ativo sem descarta-la.
Os comandos correspondentes sairam de `[project.scripts]` para que a vitrine do
projeto mostre o fluxo oficial; eles continuam acessiveis por modulo, por
exemplo:

    uv run python -m tech_challenge_fase2._historical.run_provider_real_evaluation_v4 --help

Nenhum destes modulos deve ser executado durante a demonstracao. Varios chamam
provider real quando recebem credencial, e a evidencia preservada nao deve ser
regerada: ela documenta uma chamada unica, sem retry.
"""

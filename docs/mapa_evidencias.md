# Mapa de evidências

As afirmações quantitativas devem ser auditadas nos artefatos estruturados. Documentos narrativos servem como interpretação, não como autoridade superior.

| Afirmação | Evidência prioritária | Campo ou recorte |
|---|---|---|
| Dataset possui 569 registros | `artifacts/final_evaluation/final_evaluation_plan.json` | `split.source_rows` |
| Desenvolvimento/holdout são 455/114 | `final_evaluation_plan.json` | `split.development_rows`, `split.test_rows` |
| Holdout possui 42 malignos | `final_evaluation_plan.json` | `split.test_class_counts.1` |
| Threshold permaneceu 0,5 | `final_test_results.json` | `classification_threshold` |
| Seleção não foi reaberta | `final_test_results.json` | `selection_reopened=false` |
| Não houve nova otimização na avaliação final | `final_test_results.json` | `new_optimization_performed=false` |
| Plano foi assinado antes da avaliação | `final_evaluation_plan.json` | `created_at_utc`, `signature` |
| Manifesto da Missão 4 está íntegro | `final_manifest.json` | `files`, `signature` |
| Vencedor global pré-holdout é LR da busca aleatória | `artifacts/selection/frozen_candidates.json` | `global_provisional_winner` |
| RF GA C venceu sua família por CV | `frozen_candidates.json` | `winners_by_model.random_forest` |
| KNN GA A foi serializado para a família | `frozen_candidates.json` | `winners_by_model.knn` |
| LR GA avaliado na Missão 4 veio de GA B | `final_evaluation_plan.json` | candidato `logistic_regression__ga.origin` |
| LR baseline teve FN=3 | `final_test_results.json` | candidato LR baseline, `metrics.false_negatives` |
| LR GA teve FN=1 | `final_test_results.json` | candidato LR GA, `metrics.false_negatives` |
| RF baseline/GA tiveram FN 4/3 | `final_test_results.json` | candidatos RF, `metrics.false_negatives` |
| KNN não melhorou recall/FN | `final_test_results.json` | candidatos KNN baseline/GA |
| GA e aleatória de RF têm matriz igual e AUC distinta | `final_test_results.json` | métricas RF GA/random search |
| IC95% do recall | `uncertainty_results.json` | `candidate_intervals.*.recall_malignant` |
| Delta de recall e bootstrap | `uncertainty_results.json` | `paired_baseline_vs_ga.*.bootstrap` |
| McNemar e discordantes | `uncertainty_results.json` | `paired_baseline_vs_ga.*.mcnemar` |
| Baseline histórico já registrou holdout | `artifacts/baseline_results.json` + preflight | métricas históricas e exclusão de linhagem |
| Provider oficial LLM foi mock | `artifacts/llm_evaluation/llm_evaluation_manifest.json` | `provider=fake` |
| Nenhum dado individual foi enviado | `llm_evaluation_manifest.json` | `individual_data_sent=false` |
| Factualidade foi aprovada | `factuality_report.json` | `passed=true`, `checks` |
| Safety checker foi aprovado | `safety_report.json` | `passed=true`, `violations=[]` |
| Cinco dimensões obtiveram 1,0 | `evaluation_report.json` | `dimensions`, `overall_score` |
| LLM foi idempotente | `llm_evaluation_manifest.json` | `run_identity` e hash preservado |
| Contrato LLM V2 foi aprovado offline | `artifacts/llm_contract_v2/contract_v2_manifest.json` | `status=approved`, `ready_for_real_v2_evaluation=true` |
| Provider real V2 respondeu uma vez | `artifacts/llm_evaluation_openai_v4/llm_evaluation_manifest.json` | `call_budget.scientific_main=1`, `automatic_retries=0` |
| Resposta real reproduziu todos os fatos | `artifacts/llm_evaluation_openai_v4/factuality_report.json` | `passed_checks=327`, `check_count=327` |
| Resposta real permaneceu não aprovada | `artifacts/llm_evaluation_openai_v4/llm_evaluation_manifest.json` | `status=methodologically_complete_not_approved`, `scientific_evaluation_approved=false` |
| Nenhum dado individual foi enviado ao provider real | `artifacts/llm_evaluation_openai_v4/llm_evaluation_manifest.json` | `privacy.individual_data_sent=false` |
| Classificação individual foi explicada | `artifacts/llm_individual_explanation_openai/individual_output.json` | `structured_output` |
| Caso individual veio somente do desenvolvimento | `individual_input_snapshot.json` | `source_scope=development_only`, `test_or_holdout_case_used=false` |
| Nenhuma linha ou identificador foi enviado | `privacy_report.json` | IDs, índice, target e valores brutos ausentes |
| Explicação individual reproduziu os fatos | `factuality_report.json` | `passed_checks=40`, `total_checks=40` |
| Qualidade individual passou seis dimensões | `evaluation_report.json` | `approved=true`, `overall_score=1.0` |
| Insights não decidem cuidado | `individual_output.json` | `scope=human_review_only`, `patient_care_decision=false` |
| Base textual do Módulo 3 está preparada | `individual_output.json` | `preparacao_modulo3.ready_for_future_text=true` |
| Tabela mestre não executou modelagem | `artifacts/final_summary/model_results.json` | `data_scope` |
| Figuras usam somente agregados | `reports/figures/final_presentation/figure_qa_report.json` | `source_scope` |
| Consolidação não fez treino, inferência ou deploy; provider real foi isolado | `artifacts/final_summary/final_delivery_manifest.json` | `scope_confirmations`, `supplementary_real_llm_evaluation` |

## Caminhos completos prioritários

- [`final_test_results.json`](../artifacts/final_evaluation/final_test_results.json)
- [`uncertainty_results.json`](../artifacts/final_evaluation/uncertainty_results.json)
- [`final_evaluation_plan.json`](../artifacts/final_evaluation/final_evaluation_plan.json)
- [`final_manifest.json`](../artifacts/final_evaluation/final_manifest.json)
- [`frozen_candidates.json`](../artifacts/selection/frozen_candidates.json)
- [`llm_evaluation_manifest.json`](../artifacts/llm_evaluation/llm_evaluation_manifest.json)
- [`contract_v2_manifest.json`](../artifacts/llm_contract_v2/contract_v2_manifest.json)
- [`llm_evaluation_manifest.json` V4](../artifacts/llm_evaluation_openai_v4/llm_evaluation_manifest.json)
- [`individual_output.json`](../artifacts/llm_individual_explanation_openai/individual_output.json)
- [`exemplo individual versionado`](examples/individual_explanation_v1.json)
- [`model_results.json`](../artifacts/final_summary/model_results.json)
- `artifacts/final_summary/final_delivery_manifest.json` (gerado somente após a aprovação da suíte final)

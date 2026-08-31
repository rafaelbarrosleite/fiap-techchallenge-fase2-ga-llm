"""Renderizacao do painel em um unico HTML autocontido.

O documento nao faz requisicao de rede: as figuras entram como data URI e o CSS
e o JavaScript sao embutidos. Isso mantem o painel utilizavel a partir de um
clone, offline, e mantem a reproducao independente de CDN.

Nao ha timestamp no documento. A saida precisa ser identica entre execucoes
sobre os mesmos artefatos, para que regerar o painel nao altere hashes e nao
obrigue a reselar a entrega.
"""

from __future__ import annotations

import base64
import html as html_escape
import json
from typing import Any, Iterable

from .model import FIGURES, SCALABILITY_FIGURE, DashboardData

TITULO = "Tech Challenge Fase 2 — otimização genética e explicação segura"

DISCLAIMER = (
    "Resultado acadêmico e experimental. Os modelos não foram validados para uso "
    "clínico e não devem ser usados para diagnóstico, tratamento ou decisão médica."
)

CSS = """
:root{color-scheme:light;
--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
--grid:#e1e0d9;--rule:#c3c2b7;--ring:rgba(11,11,11,0.10);
--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--crit:#d03b3b;--success-ink:#006300;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink-2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--rule:#383835;--ring:rgba(255,255,255,0.10);
--s1:#3987e5;--s2:#d95926;--good:#0ca30c;--crit:#d03b3b;--success-ink:#0ca30c;}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 72px}
header h1{font-size:26px;line-height:1.25;margin:0 0 6px;letter-spacing:-.01em}
header p.sub{margin:0;color:var(--ink-2);font-size:14px}
.disclaimer{margin:18px 0 24px;padding:12px 14px;border-radius:10px;
background:color-mix(in srgb,var(--crit) 8%,var(--surface));
border:1px solid color-mix(in srgb,var(--crit) 34%,transparent);
color:var(--ink);font-size:13.5px}
.disclaimer strong{color:var(--crit)}
nav{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:22px;
border-bottom:1px solid var(--grid);padding-bottom:0}
nav button{appearance:none;border:0;background:none;cursor:pointer;color:var(--ink-2);
font:inherit;font-size:14px;padding:9px 13px;border-bottom:2px solid transparent;
margin-bottom:-1px;border-radius:6px 6px 0 0}
nav button:hover{color:var(--ink);background:var(--surface)}
nav button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--s1);font-weight:600}
section[hidden]{display:none}
h2{font-size:19px;margin:30px 0 4px;letter-spacing:-.01em}
h2:first-child{margin-top:0}
p.lead{margin:0 0 16px;color:var(--ink-2);font-size:14px;max-width:74ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(184px,1fr));gap:12px;margin:16px 0 8px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-size:27px;font-weight:650;letter-spacing:-.02em;margin-top:5px;
font-variant-numeric:tabular-nums}
.tile .n{font-size:12.5px;color:var(--ink-2);margin-top:3px}
.tile .v .down{color:var(--success-ink)}
.tw{overflow-x:auto;margin:14px 0;border:1px solid var(--ring);border-radius:12px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--grid);white-space:nowrap}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
tr.sel td{background:color-mix(in srgb,var(--s1) 8%,transparent);font-weight:600}
figure{margin:18px 0;background:var(--surface);border:1px solid var(--ring);
border-radius:12px;padding:14px;overflow-x:auto}
figure img{display:block;width:100%;height:auto;border-radius:6px}
figcaption{margin-top:9px;font-size:12.5px;color:var(--ink-2)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;
padding:16px 18px;margin:14px 0}
.card h3{margin:0 0 8px;font-size:15px}
.card p{margin:0 0 10px;font-size:14px;color:var(--ink)}
.card p:last-child{margin-bottom:0}
ul.plain{margin:0;padding-left:19px;font-size:14px}
ul.plain li{margin-bottom:6px}
.pill{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600;
padding:3px 9px;border-radius:999px;white-space:nowrap}
.pill.ok{background:color-mix(in srgb,var(--good) 15%,var(--surface));color:var(--success-ink)}
.pill.no{background:color-mix(in srgb,var(--crit) 15%,var(--surface));color:var(--crit)}
.tag{display:inline-flex;align-items:center;font-size:12px;padding:3px 9px;border-radius:999px;
background:color-mix(in srgb,var(--muted) 14%,var(--surface));color:var(--ink-2);
border:1px solid var(--ring);white-space:nowrap}
.checks{max-height:420px;overflow-y:auto;border:1px solid var(--ring);border-radius:12px;background:var(--surface)}
.checks table{font-size:12.5px}
.checks th{position:sticky;top:0;background:var(--surface);z-index:1}
.checks td.c{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;white-space:normal}
.checks td.d{color:var(--ink-2);white-space:normal}
details{margin:10px 0}
summary{cursor:pointer;font-size:13.5px;color:var(--ink-2)}
.foot{margin-top:44px;padding-top:18px;border-top:1px solid var(--grid);
font-size:12.5px;color:var(--muted)}
@media (max-width:640px){.wrap{padding:20px 14px 56px}header h1{font-size:21px}
.tile .v{font-size:23px}}
"""

JS = """
(function(){
  var tabs=[].slice.call(document.querySelectorAll('nav button'));
  var panels=[].slice.call(document.querySelectorAll('section[data-panel]'));
  function show(id){
    tabs.forEach(function(t){t.setAttribute('aria-selected',String(t.dataset.target===id));});
    panels.forEach(function(p){p.hidden=(p.dataset.panel!==id);});
  }
  tabs.forEach(function(t){t.addEventListener('click',function(){show(t.dataset.target);});});
  show(tabs[0].dataset.target);
})();
"""


# Os artefatos usam enums estaveis em ingles porque sao contrato de dados. A
# interface e em portugues, entao a traducao acontece so na renderizacao.
ROTULOS = {
    "high": "elevada", "medium": "intermediária", "low": "baixa",
    "toward_malignant": "para a classe maligna", "toward_benign": "para a classe benigna",
    "factuality": "factualidade", "completeness": "completude", "clarity": "clareza",
    "safety": "segurança", "medical_context_relevance": "relevância médica",
    "scientific_calibration": "calibração científica",
    "logistic_regression": "Regressão Logística", "random_forest": "Random Forest",
    "knn": "KNN", "baseline": "Baseline", "ga": "GA", "random_search": "Busca aleatória",
}


def rotulo(chave: Any) -> str:
    return ROTULOS.get(str(chave), str(chave))


def esc(value: Any) -> str:
    return html_escape.escape(str(value), quote=True)


def _num(value: Any, casas: int = 4) -> str:
    try:
        return f"{float(value):.{casas}f}".replace(".", ",")
    except (TypeError, ValueError):
        return esc(value)


def _pill(ok: bool, texto_ok: str = "aprovado", texto_nao: str = "reprovado") -> str:
    classe, texto = ("ok", texto_ok) if ok else ("no", texto_nao)
    simbolo = "✓" if ok else "✕"
    return f'<span class="pill {classe}">{simbolo} {esc(texto)}</span>'


def _achado(texto: str) -> str:
    """Marcador neutro para resultado cientifico.

    Verde e vermelho pertencem a verificacao: um check confere ou diverge, uma
    garantia se manteve ou nao. Um intervalo que inclui zero, um trade-off entre
    metricas ou um ganho de CV nao confirmado sao achados que o projeto reporta
    de proposito; pinta-los de vermelho os transformaria em falhas.
    """

    return f'<span class="tag">{esc(texto)}</span>'


def _tile(rotulo: str, valor: str, nota: str = "") -> str:
    nota_html = f'<div class="n">{esc(nota)}</div>' if nota else ""
    return (
        f'<div class="tile"><div class="k">{esc(rotulo)}</div>'
        f'<div class="v">{valor}</div>{nota_html}</div>'
    )


def _figure(nome: str, legenda: str, dados: bytes) -> str:
    b64 = base64.b64encode(dados).decode("ascii")
    return (
        f'<figure><img alt="{esc(legenda)}" src="data:image/png;base64,{b64}">'
        f"<figcaption>{esc(legenda)}</figcaption></figure>"
    )


def _lista(itens: Iterable[Any]) -> str:
    linhas = "".join(f"<li>{esc(item)}</li>" for item in itens)
    return f'<ul class="plain">{linhas}</ul>'


def _painel_visao_geral(data: DashboardData) -> str:
    sel = data.selected_row
    transicoes = data.false_negative_transitions
    total_fn_base = sum(b for _, b, _ in transicoes)
    total_fn_ga = sum(g for _, _, g in transicoes)

    tiles = "".join([
        _tile("Candidato congelado", esc(sel["family_label"]),
              f'{sel["method_label"]}, escolhido antes do holdout'),
        _tile("Recall maligno no holdout", _num(sel["recall_test"]),
              f'{int(sel["true_positives"])} de 42 casos malignos'),
        _tile("Falsos negativos", f'{int(sel["false_negatives"])}',
              "no candidato congelado"),
        _tile("Falsos negativos, três famílias",
              f'{total_fn_base} <span class="down">→ {total_fn_ga}</span>',
              "baseline para GA, somados"),
    ])

    linhas = []
    for row in data.master_rows:
        classe = ' class="sel"' if row["selection_status"] == "global_selected" else ""
        linhas.append(
            f"<tr{classe}><td>{esc(row['family_label'])}</td><td>{esc(row['method_label'])}</td>"
            f'<td class="n">{_num(row["recall_test"])}</td>'
            f'<td class="n">{_num(row["f1_test"])}</td>'
            f'<td class="n">{_num(row["roc_auc_test"])}</td>'
            f'<td class="n">{int(row["false_negatives"])}</td>'
            f"<td>{esc(row['selection_status'])}</td></tr>"
        )

    # Nem toda confirmacao verdadeira e uma violacao: chamar o provider real,
    # criar configuracao de nuvem ou gerar este painel sao decisoes do projeto.
    # Colorir todo True de vermelho faria realizacoes legitimas parecerem falhas.
    GARANTIAS = {
        "new_training_performed", "new_optimization_performed",
        "new_holdout_inference_performed", "threshold_changed", "selection_reopened",
        "raw_individual_record_sent_to_llm", "patient_identifier_sent_to_llm",
        "http_api_created", "cloud_resources_provisioned", "deploy_performed",
    }
    confirmacoes = data.delivery_manifest.get("scope_confirmations", {})
    garantias = "".join(
        f"<tr><td>{esc(k)}</td><td>{_pill(v is False, 'não ocorreu', 'ocorreu')}</td></tr>"
        for k, v in confirmacoes.items() if k in GARANTIAS
    )
    fatos = "".join(
        f"<tr><td>{esc(k)}</td><td>{'sim' if v else 'não'}</td></tr>"
        for k, v in confirmacoes.items() if k not in GARANTIAS
    )

    return f"""
<h2>Resultado confirmatório</h2>
<p class="lead">A seleção ocorreu apenas nos 455 registros de desenvolvimento. Os candidatos
foram congelados antes do holdout de 114 registros, e o teste final não alterou
hiperparâmetros, limiar nem modelo selecionado.</p>
<div class="tiles">{tiles}</div>

<h2>Tabela mestre</h2>
<p class="lead">As nove origens avaliadas no mesmo holdout, com limiar fixo de 0,5.
A linha destacada é o vencedor global.</p>
<div class="tw"><table>
<thead><tr><th>Família</th><th>Método</th><th class="n">Recall</th><th class="n">F1</th>
<th class="n">ROC-AUC</th><th class="n">FN</th><th>Situação</th></tr></thead>
<tbody>{''.join(linhas)}</tbody></table></div>

<h2>Garantias de escopo</h2>
<p class="lead">O que a consolidação declara explicitamente <em>não</em> ter feito.
Cada linha é uma barreira: se qualquer uma tivesse ocorrido, o resultado
confirmatório deixaria de valer.</p>
<div class="tw"><table><tbody>{garantias}</tbody></table></div>

<h2>Decisões registradas</h2>
<p class="lead">Fatos do projeto, não violações. Chamar o provider real, enviar uma
representação desidentificada, versionar configuração de nuvem e gerar este painel
foram escolhas deliberadas e documentadas.</p>
<div class="tw"><table><tbody>{fatos}</tbody></table></div>
"""


def _painel_genetico(data: DashboardData) -> str:
    figuras = "".join(
        _figure(nome, legenda, data.figures[nome]) for nome, legenda in FIGURES
    )
    transicoes = "".join(
        f"<tr><td>{esc(fam)}</td><td class='n'>{base}</td><td class='n'>{ga}</td>"
        f"<td class='n'>{'−' if ga < base else '='}{abs(base - ga) or ''}</td></tr>"
        for fam, base, ga in data.false_negative_transitions
    )
    return f"""
<h2>Otimização por algoritmo genético</h2>
<p class="lead">Codificação tipada por família, seleção por torneio, crossover uniforme,
mutação por tipo de gene, reparação, elitismo e cache. O fitness combina
0,60 × recall + 0,25 × F1 + 0,15 × ROC-AUC − 0,10 × desvio-padrão do recall,
medido em cinco dobras apenas no desenvolvimento.</p>

<h2>Falsos negativos: baseline para GA</h2>
<p class="lead">O objetivo prioritário generalizou para duas das três famílias.
No KNN, o ganho observado em validação cruzada não se confirmou.</p>
<div class="tw"><table>
<thead><tr><th>Família</th><th class="n">Baseline</th><th class="n">GA</th><th class="n">Δ</th></tr></thead>
<tbody>{transicoes}</tbody></table></div>

<h2>Evidência visual</h2>
{figuras}
"""


def _painel_llm_agregada(data: DashboardData) -> str:
    saida = data.llm_output["structured_output"]
    aval = data.llm_evaluation
    fact = data.llm_factuality
    checks = fact["checks"]
    aprovados = sum(1 for c in checks if c["passed"])

    tiles = "".join([
        _tile("Verificações factuais", f"{aprovados}/{len(checks)}",
              "cada número recalculado do artefato congelado"),
        _tile("Nota geral", _num(aval["overall_score"], 2), "cinco dimensões determinísticas"),
        _tile("Provider", esc(data.llm_output["provider"]),
              esc(data.llm_output["model"])),
        _tile("Juiz LLM usado", "não" if not aval.get("llm_judge_used") else "sim",
              "as barreiras são código, não outro modelo"),
    ])

    def _linhas_metodo(comp: dict[str, Any]) -> str:
        linhas = ""
        for chave in ("baseline", "ga", "random_search"):
            m = comp.get(chave)
            if not m:
                continue
            linhas += (
                f"<tr><td>{esc(rotulo(chave))}</td>"
                f'<td class="n">{_num(m["recall_malignant"])}</td>'
                f'<td class="n">{_num(m["f1_malignant"])}</td>'
                f'<td class="n">{_num(m["roc_auc"])}</td>'
                f'<td class="n">{int(m["false_negatives"])}</td></tr>'
            )
        return linhas

    comparacoes = "".join(
        f'<div class="card"><h3>{esc(rotulo(c["model"]))}</h3>'
        f'<p>{esc(c["interpretation"])}</p>'
        f'<p>{_achado("ganho de CV confirmado no holdout" if c["cv_gain_confirmed_on_holdout"] else "ganho de CV não confirmado")} '
        f'{_achado("há trade-off entre métricas" if c["tradeoff_present"] else "sem trade-off entre métricas")}</p>'
        f'<div class="tw"><table><thead><tr><th>Método</th><th class="n">Recall</th>'
        f'<th class="n">F1</th><th class="n">ROC-AUC</th><th class="n">FN</th></tr></thead>'
        f"<tbody>{_linhas_metodo(c)}</tbody></table></div></div>"
        for c in saida.get("comparacao_modelos", [])
    )

    def _ic(intervalo: dict[str, Any]) -> str:
        return f'[{_num(intervalo["lower"])}; {_num(intervalo["upper"])}]'

    incertezas = "".join(
        f'<tr><td>{esc(rotulo(i["model"]))}</td>'
        f"<td>{_ic(i['baseline_recall_ci'])}</td>"
        f"<td>{_ic(i['ga_recall_ci'])}</td>"
        f'<td class="n">{_num(i["delta_recall"])}</td>'
        f"<td>{_ic(i['delta_recall_ci'])}</td>"
        f'<td>{_achado("inclui zero" if i["delta_ci_includes_zero"] else "não inclui zero")}</td>'
        f'<td class="n">{_num(i["mcnemar_p_value"], 3)}</td></tr>'
        for i in saida.get("incerteza_por_modelo", [])
    )

    linhas_check = "".join(
        f'<tr><td class="c">{esc(c["check"])}</td>'
        f'<td>{_pill(c["passed"], "confere", "diverge")}</td>'
        f'<td class="d">{esc(c.get("detail", ""))}</td></tr>'
        for c in checks
    )

    seguranca = data.llm_safety
    dimensoes = "".join(
        f"<tr><td>{esc(rotulo(nome))}</td><td>{_pill(bool(valor.get('passed')))}</td>"
        f'<td class="n">{_num(valor.get("score", 0), 2)}</td></tr>'
        for nome, valor in aval.get("dimensions", {}).items()
    )

    return f"""
<h2>Explicação dos resultados agregados</h2>
<p class="lead">A LLM recebe apenas resultados experimentais. O contrato rejeita registros,
features, índices, diagnósticos e probabilidades individuais.</p>
<div class="tiles">{tiles}</div>

<div class="card"><h3>Resumo executivo gerado</h3><p>{esc(saida['resumo_executivo'])}</p></div>
<div class="card"><h3>Interpretação do algoritmo genético</h3><p>{esc(saida['interpretacao_ga'])}</p></div>
<div class="card"><h3>Incerteza estatística</h3><p>{esc(saida['incerteza_estatistica'])}</p></div>

<h2>Comparação por família</h2>
<p class="lead">Os três métodos avaliados no mesmo holdout, com a leitura que a
explicação deu a cada família.</p>
{comparacoes}

<h2>Incerteza por família</h2>
<p class="lead">Intervalos de Wilson a 95%, delta pareado por bootstrap e McNemar exato.
Um intervalo que toca zero não demonstra diferença; um p alto não prova igualdade.</p>
<div class="tw"><table>
<thead><tr><th>Família</th><th>IC do baseline</th><th>IC do GA</th>
<th class="n">Δ recall</th><th>IC do Δ</th><th>Leitura</th><th class="n">McNemar p</th></tr></thead>
<tbody>{incertezas}</tbody></table></div>

<h2>Limitações declaradas pela própria explicação</h2>
{_lista(saida.get('limitacoes', []))}

<div class="card"><h3>Conclusão</h3><p>{esc(saida['conclusao'])}</p></div>

<h2>Verificação independente</h2>
<p class="lead">O provider gera; código determinístico decide se a saída pode ser aprovada.
Cada linha abaixo recalcula um valor a partir do artefato congelado e o compara
com o que a explicação afirmou. Nenhum outro modelo participa do julgamento.</p>
<div class="tiles">
{_tile("Segurança", "sem violações" if seguranca.get("passed") else "violações", "diagnóstico, tratamento, recomendação e certeza indevida")}
{_tile("Números não autorizados", str(len(fact.get("unexpected_text_numbers", []))), "presentes na narrativa mas ausentes do contrato")}
</div>
<div class="tw"><table>
<thead><tr><th>Dimensão</th><th>Situação</th><th class="n">Nota</th></tr></thead>
<tbody>{dimensoes}</tbody></table></div>
<details open><summary>As {len(checks)} verificações factuais, uma a uma</summary>
<div class="checks"><table>
<thead><tr><th>Verificação</th><th>Resultado</th><th>Valor recalculado</th></tr></thead>
<tbody>{linhas_check}</tbody></table></div></details>
"""


def _painel_individual(data: DashboardData) -> str:
    saida = data.individual_output["structured_output"]
    aval = data.individual_evaluation
    classif = saida["classificacao_do_modelo"]
    fatores = saida["fatores_explicativos"]

    tiles = "".join([
        _tile("Referência do caso", esc(saida["case_reference"]),
              "identificador opaco; a linha original não chega à LLM"),
        _tile("Probabilidade estimada", _num(classif["probability_malignant"], 5),
              f'limiar fixo de {_num(classif["classification_threshold"], 1)}'),
        _tile("Predição é diagnóstico?",
              "não" if saida.get("predicao_nao_e_diagnostico") is True else "campo ausente",
              "o contrato exige a afirmação explícita"),
        _tile("Uso clínico autorizado",
              "não" if saida.get("uso_clinico_autorizado") is False else "sim",
              "estruturalmente proibido"),
    ])

    linhas_fator = "".join(
        f'<tr><td class="n">{int(f["rank"])}</td><td>{esc(f["display_name"])}</td>'
        f'<td>{esc(rotulo(f["observed_band"]))}</td>'
        f'<td>{esc(rotulo(f["influence_direction"]))}</td>'
        f'<td class="n">{_num(f["relative_importance_percent"], 2)}%</td></tr>'
        for f in fatores
    )
    fatores_texto = "".join(
        f'<div class="card"><h3>{int(f["rank"])}. {esc(f["display_name"])}</h3>'
        f'<p>{esc(f["explanation"])}</p></div>'
        for f in fatores
    )
    insights = "".join(
        f'<div class="card"><h3>{esc(i["action"])}</h3><p>{esc(i["rationale"])}</p>'
        f'<p>{_pill(i["scope"] == "human_review_only", "escopo: revisão humana", "escopo aberto")} '
        f'{_pill(i["patient_care_decision"] is False, "não é decisão de cuidado", "decisão de cuidado")}</p></div>'
        for i in saida.get("insights_acionaveis_para_medicos", [])
    )

    modulo3 = saida.get("preparacao_modulo3", {})
    dimensoes = "".join(
        f"<tr><td>{esc(rotulo(nome))}</td><td>{_pill(bool(valor.get('passed')))}</td>"
        f'<td class="n">{_num(valor.get("score", 0), 2)}</td></tr>'
        for nome, valor in aval.get("dimensions", {}).items()
    )

    return f"""
<h2>Explicação de uma classificação individual</h2>
<p class="lead">O enunciado pede explicações dos diagnósticos produzidos pelos modelos, não
apenas dos agregados. O contrato 3.0 explica um caso do desenvolvimento usando o
pipeline congelado, sem novo treino e sem tocar o holdout.</p>
<div class="tiles">{tiles}</div>

<div class="card"><h3>Resumo</h3><p>{esc(saida['resumo_executivo'])}</p></div>
<div class="card"><h3>Como a classe foi definida</h3><p>{esc(classif['interpretation'])}</p></div>

<h2>O que a LLM recebe — e o que não recebe</h2>
<p class="lead">A LLM recebe classe, probabilidade, limiar e cinco sinais com faixa,
direção e importância relativa. Não recebe identificador, índice, diagnóstico real,
alvo nem os trinta valores medidos. Nomear o sinal é o que torna a explicação útil;
o valor medido é o que a reconstruiria, e ele não é transmitido.</p>
<div class="tw"><table>
<thead><tr><th class="n">#</th><th>Sinal</th><th>Faixa</th><th>Direção</th>
<th class="n">Importância</th></tr></thead>
<tbody>{linhas_fator}</tbody></table></div>

<h2>Os cinco fatores em linguagem natural</h2>
{fatores_texto}

<h2>Insights acionáveis</h2>
<p class="lead">Cada ação é estruturalmente limitada a revisão humana e não pode
representar decisão de cuidado.</p>
{insights}

<h2>Limitações declaradas</h2>
{_lista(saida.get('limitacoes', []))}

<h2>Preparação para o Módulo 3</h2>
<div class="card">
<p>{esc(modulo3.get('explanation', ''))}</p>
<p>{_pill(bool(modulo3.get('ready_for_future_text')), 'campos textuais preparados', 'não preparado')}
{_pill(modulo3.get('current_text_data_used') is False, 'nenhum texto clínico usado agora', 'texto em uso')}</p>
</div>

<h2>Avaliação da qualidade</h2>
<div class="tw"><table>
<thead><tr><th>Dimensão</th><th>Situação</th><th class="n">Nota</th></tr></thead>
<tbody>{dimensoes}</tbody></table></div>
"""


def _painel_escalabilidade(data: DashboardData) -> str:
    if data.scalability is None:
        return (
            '<h2>Escalabilidade automática</h2><p class="lead">O relatório de '
            "escalabilidade não está presente neste clone. Execute "
            "<code>uv run run-load-benchmark</code> para gerá-lo.</p>"
        )
    rel = data.scalability
    cenarios = {c["label"]: c for c in rel["scenarios"]}
    fixo, escala = cenarios["pool_fixo_minimo"], cenarios["pool_autoescalavel"]
    comp = rel["comparison"]
    amb = rel["environment"]

    tiles = "".join([
        _tile("Redução da latência p95", f'{_num(comp["p95_latency_reduction_factor"], 2)}x',
              "pool autoescalável contra pool fixo mínimo"),
        _tile("Ganho de vazão", f'{_num(comp["throughput_gain_factor"], 2)}x',
              "mesma sequência de chegadas"),
        _tile("Teto de réplicas", str(rel["policy"]["max_workers"]),
              f'em {amb["available_cpus"]} CPUs disponíveis'),
        _tile("Threads de BLAS por worker", esc(amb["blas_threads_per_worker"]),
              "o paralelismo vem das réplicas"),
    ])

    comparativo = f"""
<tr><td>Pool fixo mínimo</td><td class="n">{_num(fixo['latency']['p95_ms'], 1)}</td>
<td class="n">{_num(fixo['latency']['p99_ms'], 1)}</td>
<td class="n">{_num(fixo['throughput_requests_per_second'], 1)}</td>
<td class="n">{fixo['max_workers_used']}</td><td class="n">{fixo['scaling_events']}</td></tr>
<tr><td>Pool autoescalável</td><td class="n">{_num(escala['latency']['p95_ms'], 1)}</td>
<td class="n">{_num(escala['latency']['p99_ms'], 1)}</td>
<td class="n">{_num(escala['throughput_requests_per_second'], 1)}</td>
<td class="n">{escala['max_workers_used']}</td><td class="n">{escala['scaling_events']}</td></tr>"""

    sweep = "".join(
        f'<tr><td class="n">{s["batch_size"]}</td>'
        f'<td class="n">{_num(s["milliseconds_per_request_serial"], 1)}</td>'
        f'<td class="n">{_num(s["speedup"], 2)}x</td>'
        f'<td>{_achado("compensa" if s["speedup"] >= 1.0 else "não compensa")}</td></tr>'
        for s in rel["batch_size_sweep"]
    )

    nome, legenda = SCALABILITY_FIGURE
    return f"""
<h2>Escalabilidade automática e monitoramento</h2>
<p class="lead">A camada de serviço executa o modelo congelado sob demanda variável.
A política de dimensionamento é uma função pura do backlog, com histerese e cooldown.
Nada aqui treina, reabre seleção ou altera o limiar.</p>
<div class="tiles">{tiles}</div>

<h2>Mesma demanda, duas configurações</h2>
<div class="tw"><table>
<thead><tr><th>Cenário</th><th class="n">p95 (ms)</th><th class="n">p99 (ms)</th>
<th class="n">req/s</th><th class="n">Workers</th><th class="n">Trocas</th></tr></thead>
<tbody>{comparativo}</tbody></table></div>

<h2>Escalar réplicas tem um limiar</h2>
<p class="lead">Abaixo de cerca de 2 ms por pedido, o despacho custa mais que o trabalho e
adicionar réplicas piora o desempenho. A varredura inteira ficou na evidência em vez
de se escolher o tamanho de lote que favorecia a conclusão.</p>
<div class="tw"><table>
<thead><tr><th class="n">Registros por pedido</th><th class="n">ms por pedido</th>
<th class="n">Aceleração</th><th>Leitura</th></tr></thead>
<tbody>{sweep}</tbody></table></div>

{_figure(nome, legenda, data.figures[nome])}

<p class="lead">A medição depende do hardware e não deve ser citada como característica
do modelo. O relatório registra isso explicitamente.</p>
"""


def render_dashboard(data: DashboardData) -> str:
    """Monta o documento completo, sem timestamp e sem recurso externo."""

    paineis = [
        ("visao", "Visão geral", _painel_visao_geral(data)),
        ("genetico", "Algoritmo genético", _painel_genetico(data)),
        ("llm", "LLM agregada", _painel_llm_agregada(data)),
        ("individual", "LLM individual", _painel_individual(data)),
        ("escala", "Escalabilidade", _painel_escalabilidade(data)),
    ]
    abas = "".join(
        f'<button role="tab" data-target="{chave}" aria-selected="false">{esc(rotulo)}</button>'
        for chave, rotulo, _ in paineis
    )
    secoes = "".join(
        f'<section data-panel="{chave}" hidden>{conteudo}</section>'
        for chave, _, conteudo in paineis
    )
    sel = data.selected_row
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(TITULO)}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<header>
<h1>{esc(TITULO)}</h1>
<p class="sub">Painel somente leitura sobre artefatos assinados. Não treina, não reabre
seleção, não altera o limiar e não faz chamada de rede.</p>
</header>
<div class="disclaimer"><strong>Aviso.</strong> {esc(DISCLAIMER)}</div>
<nav role="tablist">{abas}</nav>
{secoes}
<p class="foot">Candidato congelado: {esc(sel['family_label'])} ({esc(sel['method_label'])}).
Gerado por <code>uv run build-dashboard</code> a partir dos artefatos versionados.
Este documento não contém dado de paciente.</p>
</div>
<script>{JS}</script>
</body></html>
"""

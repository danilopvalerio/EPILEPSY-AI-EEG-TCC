# Pipeline de Predição de Crises Epilépticas em EEG — Documentação

Este documento explica o pipeline completo, dividido em **4 notebooks**, e
detalha as decisões de método — com ênfase nas partes que costumam confundir.
A ideia central do trabalho é dupla: medir **quanto desempenho se perde ao
reduzir o número de eletrodos** (rumo a um dispositivo vestível) e verificar se
**misturar vários datasets no treino melhora a generalização**.

---

## 1. Visão geral e ordem de execução

Os notebooks rodam em sequência. Cada um é **independente** (redefine a
configuração no topo) e se comunica com o seguinte por arquivos salvos em
`data/`. Isso existe porque cada notebook é um kernel separado: não há memória
compartilhada entre eles, então o estado passa pelo disco.

| Notebook                             | O que faz                                                            | Entrada      | Saída                       |
| ------------------------------------ | -------------------------------------------------------------------- | ------------ | --------------------------- |
| **NB1** Download & Pré-processamento | baixa os 4 datasets, lê anotações, reamostra, filtra, rotula         | EDFs da web  | `data/level1_signals/` (L1) |
| **NB2** Janelas, Níveis & Features   | janela de 4 s, extrai features por região, define os níveis de canal | L1           | `data/features/<nível>/`    |
| **NB3** Experimentos                 | LOSO nos cenários A/B/C, por nível e modelo                          | features     | `data/results/results.json` |
| **NB4** Métricas & Gráficos          | tabelas, curva de degradação, comparativos                           | results.json | gráficos + CSVs             |

---

## 2. Os quatro datasets

| Dataset            | Origem        | fs original | Montagem                     | Papel no estudo         |
| ------------------ | ------------- | ----------- | ---------------------------- | ----------------------- |
| **CHB-MIT**        | PhysioNet     | 256 Hz      | bipolar (double-banana)      | clínico, alta densidade |
| **Siena**          | PhysioNet     | 512 Hz      | referencial 10-20            | clínico, alta densidade |
| **Mendeley (AUB)** | Mendeley Data | 500 Hz      | referencial 10-20 (19–21 ch) | clínico, alta densidade |
| **SeizeIT2**       | OpenNeuro     | 256 Hz      | behind-the-ear (2 ch)        | **alvo vestível**       |

Cada dataset traz as crises de forma diferente: CHB-MIT num `summary.txt`, Siena
numa `Seizures-list-PNxx.txt`, SeizeIT2 em `_events.tsv` (BIDS) e Mendeley numa
**planilha XLSX**. O NB1 tem um parser para cada um.

Para baixar menos, o pipeline só busca arquivos **com crise**: usa os índices de
anotação para descobrir quais gravações têm crise antes de baixar os EDFs
(que são grandes). No SeizeIT2 isso é crítico, porque há EDFs de até 18 horas.

### 2.1 Detalhe complicado: o XLSX do Mendeley guarda tempo como fração de dia

Na planilha do Mendeley, os tempos (onset, file onset) não estão em segundos:
estão como **fração de um dia** (um número entre 0 e 1, onde 0,5 = meio-dia). O
parser multiplica por `24 × 3600` para obter segundos e subtrai o início do
arquivo para obter o tempo **relativo** ao começo da gravação. A duração da
crise vem como texto ("1 min 16 sec") e é extraída por expressão regular.

---

## 3. A decisão mais importante: por que agregar por REGIÃO e não por eletrodo

Este é o ponto que mais gera confusão, então vale detalhar.

### 3.1 O problema

Para treinar um modelo, **todos os exemplos precisam ter o mesmo número de
features**. Mas CHB-MIT tem ~23 canais, Siena ~35, Mendeley ~19–21 e SeizeIT2
apenas 2. Não dá para simplesmente concatenar as features de todos os canais,
porque os vetores teriam tamanhos diferentes e não poderiam ser empilhados.

### 3.2 Por que "usar os mesmos eletrodos" não funciona aqui

A ideia natural seria pegar os eletrodos em comum entre os datasets. Isso falha
por dois motivos:

Primeiro, o **SeizeIT2 só tem 2 canais** (behind-the-ear). Se o conjunto comum
for limitado ao que ele tem, sobrariam 2 eletrodos para todo mundo, jogando fora
quase toda a riqueza espacial dos datasets clínicos.

Segundo, e mais sutil, as **montagens são incompatíveis**. O CHB-MIT é bipolar:
seus canais são _pares_ como `FP1-F7`, `F7-T7`. Ele não tem um canal "F3"
isolado. Já o Siena e o Mendeley são referenciais e têm eletrodos individuais.
O SeizeIT2 nem usa o sistema 10-20 — é behind-the-ear. Então "selecionar o
eletrodo F3 em todos" simplesmente não está bem definido.

### 3.3 A solução: pooling por região cerebral

Em vez de trabalhar canal a canal, agrupamos os canais por **região do cérebro**
e calculamos a média das features daquela região:

```
frontal   ← Fp1, Fp2, F3, F4, F7, F8, Fz ...
temporal  ← T3, T4, T5, T6 (e behind-the-ear do SeizeIT2)
central   ← C3, C4, Cz ...
parietal  ← P3, P4, Pz ...
occipital ← O1, O2 ...
```

Para cada janela e cada região, pegamos as features dos canais daquela região e
tiramos a média. Resultado: **16 features × número de regiões**, sempre do mesmo
tamanho, qualquer que seja o dataset ou o número de canais.

Isso resolve os três problemas de uma vez:

1. **Tamanho fixo** → datasets diferentes geram o mesmo vetor e podem ser
   misturados no treino.
2. **Funciona em qualquer montagem** → um canal bipolar `F7-T7` é atribuído à
   região pelo seu primeiro nó (`F7` → frontal); um referencial `F3` igual; o
   behind-the-ear do SeizeIT2 é mapeado para temporal.
3. **Preserva interpretabilidade** → como sabemos qual região é qual, a etapa de
   SHAP poderá dizer "a região temporal foi a mais importante", o que a média
   global de todos os canais (abordagem antiga) tornava impossível.

### 3.4 Por que a média global anterior era ruim

A versão anterior tirava a média de **todos** os canais juntos, gerando 32
features fixas. O problema: se a crise começa concentrada em poucos canais (foco
focal), a média de todos dilui esse sinal, e perde-se a noção de _onde_ a crise
aconteceu. O pooling por região conserta isso porque mantém as regiões
separadas — uma crise temporal aparece na região temporal, não some na média.

---

## 4. Os níveis de redução de canais

Os níveis de canal são definidos como **quantas regiões** entram. Isso transforma
"reduzir eletrodos" numa escala limpa e comparável.

| Nível  | Regiões                        | Análogo clínico                  | Datasets                 |
| ------ | ------------------------------ | -------------------------------- | ------------------------ |
| **R5** | 5 (todas)                      | EEG hospitalar completo (~19 ch) | CHB-MIT, Siena, Mendeley |
| **R3** | 3 (frontal, temporal, central) | reduzido (~8 ch)                 | CHB-MIT, Siena, Mendeley |
| **R2** | 2 (frontal, temporal)          | vestível (~4 ch)                 | CHB-MIT, Siena, Mendeley |
| **R1** | 1 (temporal)                   | ultra-compacto / vestível real   | **+ SeizeIT2**           |

O **SeizeIT2 só entra em R1**, porque seus eletrodos behind-the-ear correspondem
aproximadamente à região temporal — é o único nível onde ele se encaixa
fisicamente. Os datasets clínicos são "rebaixados" progressivamente até esse
mesmo formato.

A pergunta central do trabalho fica: **quanto desempenho se perde ao migrar de
R5 (hospital) para R1 (vestível)?** A resposta é a _curva de degradação_, gerada
no NB4.

---

## 5. Os dois eixos do estudo

O trabalho cruza dois eixos independentes. Confundi-los é fácil, então segue a
separação clara.

**Eixo 1 — Níveis de canal** (quantas regiões): R5 → R3 → R2 → R1. Responde à
pergunta da viabilidade vestível.

**Eixo 2 — Cenários de treino/teste** (quem treina, quem testa):

- **A_intra** — treina e testa no **mesmo** dataset.
- **B_cross** — treina em **todo** o dataset X, testa em cada paciente do
  dataset Y.
- **C_mix** — treina na **união** dos outros datasets, testa em cada paciente do
  dataset deixado de fora.

Comparar **A vs C** responde à segunda pergunta do trabalho: _misturar datasets
no treino melhora a generalização?_ Se C for muito melhor que A, misturar ajuda;
se forem parecidos, o modelo já generaliza sozinho — o que também é um resultado
forte para defender um sistema vestível universal.

Os dois eixos compartilham a mesma extração de features (pooling por região),
então rodar tudo não duplica trabalho: é a mesma matriz de features avaliada em
combinações diferentes de "quem treina/testa" e "quantas regiões".

---

## 6. Como o LOSO se encaixa

LOSO (Leave-One-Subject-Out) **não é um cenário**; é o _método de avaliação_
dentro de cada cenário. A unidade deixada de fora é sempre **um paciente**.

Em **A_intra** (mesmo dataset):

```
Fold 1 → testa chb01 | treina chb02, chb03, ...
Fold 2 → testa chb02 | treina chb01, chb03, ...
```

Em **C_mix** (datasets misturados), o paciente deixado de fora é cada paciente do
dataset de teste, e o treino inclui todos os outros datasets inteiros:

```
treina CHB-MIT + Siena + Mendeley → testa sub-001 (SeizeIT2)
treina CHB-MIT + Siena + Mendeley → testa sub-002 (SeizeIT2)
```

Em todos os casos, o LOSO garante que o paciente testado **nunca** esteve no
treino. O que muda entre cenários é apenas _quem está no conjunto de treino_.

---

## 7. Rotulagem em 4 classes (e a margem "unknown")

Cada amostra do sinal recebe um rótulo: **interictal** (longe de crise),
**pré-ictal** (10 min antes), **ictal** (a crise) ou **pós-ictal** (10 min
depois). Existe ainda uma classe **unknown** (-1) que é uma _margem de
segurança_ entre o interictal "limpo" e o início do pré-ictal.

Por que a margem existe: sem ela, amostras logo antes do pré-ictal seriam
rotuladas como interictal, mas poderiam já conter sinais precoces de crise. Isso
confundiria o modelo. As amostras unknown são **descartadas** no janelamento, e
não entram em treino nem teste. Como `IGAP_SEC` e `PRE_SEC` valem ambos 10 min, é
normal que a fatia unknown tenha o mesmo tamanho do pré-ictal.

---

## 8. Janelamento

O sinal contínuo é recortado em **janelas de 4 segundos com 50% de
sobreposição**. Cada janela recebe o rótulo da classe majoritária dentro dela
(com prioridade para ictal). Janelas que tocam a zona unknown são descartadas.

O desbalanceamento aqui é enorme e **esperado**: um paciente passa a maior parte
do tempo sem crise. É comum ver proporções como 40:1 entre interictal e ictal.
Isso é a realidade clínica, não um defeito — e é tratado no passo seguinte.

---

## 9. Undersampling — só no treino, nunca no teste

O ponto-chave: o **treino** é balanceado, mas o **teste usa o paciente inteiro**,
com a distribuição real.

No treino, mantemos todas as janelas de evento (pré-ictal, ictal, pós-ictal) e
subamostramos o interictal até no máximo **1:3** (interictal:eventos). Isso evita
que o modelo aprenda a simplesmente chutar "interictal" sempre.

No teste, **não** balanceamos: avaliamos na distribuição natural. Se
balanceássemos o teste, as métricas ficariam infladas e não refletiriam o
desempenho clínico real (num cenário real, 95% do tempo é interictal).

Por isso o NB3 faz **dupla avaliação**: a _Realista_ (paciente inteiro, reflete a
prática) e a _Balanceada_ (teste subamostrado, mostra o teto de desempenho em
condições favoráveis).

---

## 10. O cap de interictal no SeizeIT2 (e um efeito colateral honesto)

O SeizeIT2 gera mais de 100 mil janelas por paciente, sendo 95%+ interictal.
Extrair features de tudo isso é caro e desnecessário. Então, **antes de extrair
features**, limitamos o interictal a no máximo **10:1** em relação aos eventos
(`MAX_INTER_RATIO = 10`).

O que é preservado: **todas** as janelas de evento e o **span temporal** da
gravação (mantemos a primeira e a última janela interictal). Isso é importante
porque as métricas por evento (sensibilidade, FAR/hora, lead time) dependem só da
linha do tempo dos eventos, que fica intacta — em especial o denominador do
FAR (horas totais de gravação) permanece exato.

**Efeito colateral documentado:** ao desbastar o interictal, a votação deslizante
de 10 janelas passa a cobrir um intervalo real maior nessas regiões, o que torna
a contagem de **falsos alarmes ligeiramente conservadora** (FAR tende a ser
menor) no SeizeIT2. É um efeito aceitável e registrado — a alternativa (não
aplicar o cap) é computacionalmente inviável ao escalar para vários pacientes.

---

## 11. As features (16 por canal/região)

Para cada canal, calculamos 16 features handcrafted, em três grupos:

**Temporais (6):** desvio padrão, variância, RMS, line length, Hjorth Activity,
Hjorth Mobility.

**Espectrais (6):** potência nas bandas δ, θ, α, β, γ (via Welch) e entropia
espectral.

**DWT (4):** energia das 4 sub-bandas de detalhe da wavelet db4.

Removemos features que vimos serem o gargalo computacional e pouco
discriminativas (Sample Entropy era O(N²) e travava; Hurst custava caro). No
NB3, o `SelectKBest` com _mutual information_ ainda seleciona as 20 features mais
discriminativas (ajustado só no treino), e a etapa futura de SHAP confirmará
empiricamente quais importam.

---

## 12. Predição por evento: votação, refratário e métricas clínicas

Métricas por janela (acurácia, F1) não bastam para uso clínico. O que importa é:
a crise foi **prevista com antecedência**, e **quantos alarmes falsos** o sistema
dispara.

**Votação deslizante:** sobre as predições janela a janela, uma janela de 10
predições dispara alarme quando ≥ 7 delas são pré-ictal. Isso filtra ruído — um
único erro isolado não vira alarme.

**Refratário:** após um alarme, há 5 minutos de silêncio para não contar vários
alarmes da mesma crise.

**Métricas por evento:**

- _Sensibilidade por evento_ — fração de crises previstas com antecedência.
- _FAR/hora_ — alarmes falsos por hora de gravação.
- _Lead time_ — com quantos segundos de antecedência o alarme soou (só de crises
  efetivamente previstas).

---

## 13. O resultado principal: a curva de degradação

O NB4 plota o desempenho (sensibilidade por evento, FAR/hora, sensibilidade
pré-ictal) **em função do número de regiões**: R5 (5) → R3 (3) → R2 (2) → R1 (1).

A queda de R5 para R1 quantifica **quanto se perde ao migrar do EEG hospitalar
para o formato de um vestível**. Se em R1 a sensibilidade ainda for clinicamente
aceitável, isso sustenta a viabilidade de um dispositivo vestível. Essa queda
quantificada não é uma limitação a esconder — é a contribuição do trabalho.

### 13.1 "E se eu remover justamente a região do foco da crise?"

Esse é o risco real da redução de canais, e ele é parte do que se _mede_. Crises
focais começam numa região específica; se o nível reduzido não cobre essa região,
o modelo pode falhar naquele paciente. Em vez de tratar isso como um problema a
resolver, o trabalho o **quantifica**: a planilha do Mendeley traz o foco de cada
paciente (ex.: "Right temporal", "F4/F8", "T3-C3"), permitindo cruzar o foco com
quais regiões cada nível preserva. O achado clínico — que a degradação depende de
onde fica o foco — sugere que um vestível ideal deve cobrir as regiões mais comuns
de foco (predominantemente temporal e frontal).

---

## 14. Interpretabilidade (próximo passo)

Sobre o melhor modelo, aplica-se SHAP para descobrir quais features e quais
**regiões** mais pesam na predição. Como o pooling por região mantém a identidade
espacial, o SHAP fala diretamente em termos de região cerebral e banda de
frequência — exatamente as unidades das features. Isso responde: quais regiões e
bandas marcam o estado pré-ictal. Essa etapa só é possível porque abandonamos a
média global de canais em favor do pooling por região.

---

## 15. O que conferir antes de rodar

Alguns pontos dependem dos seus dados e merecem verificação:

A lista `PATIENTS['Mendeley']` usa `p10`–`p14` como exemplo; confirme os nomes
reais dos pacientes que você baixou.

O **inventário de canais** no NB1 imprime os nomes de canal de cada dataset e a
região inferida. Se algum eletrodo aparecer sem região, ajuste o mapa
`ELECTRODE_REGION` na configuração. Em montagens bipolares, um eletrodo occipital
pode aparecer só como segundo nó de um par (ex.: `P3-O1`), e o mapeamento usa o
primeiro nó — vale conferir se a região occipital fica representada como você
espera no CHB-MIT.

Se você já tinha rodado a extração antes, **apague `data/features/`** ao mudar a
configuração de features ou o cap, para regerar do zero.

---

## 16. Limitações registradas

O pooling por região, embora resolva a incompatibilidade de montagens, perde
resolução **dentro** de cada região (vários canais viram uma média). Em montagens
bipolares, a atribuição de região pelo primeiro nó é uma aproximação. E a
comparação no nível R1 cruza montagens fisicamente diferentes (clínico reduzido a
temporal × behind-the-ear do SeizeIT2), o que é, em si, parte do que se investiga
— e deve ser apresentado como tal.

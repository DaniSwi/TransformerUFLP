# Aprendizaje por imitación con Transformer para el UFLP

Redes neuronales que aprenden a construir soluciones del **Uncapacitated Facility Location
Problem** imitando las decisiones de un algoritmo de optimización tradicional.

El proyecto recorre tres arquitecturas sobre el mismo problema y los mismos datos: una MLP
sobre el estado aplanado, un scorer equivariante a permutaciones, y un transformer
encoder-decoder con puntero. Cada una corrige una limitación concreta de la anterior.

**Integrantes:** Carlos Abarza · Daniel Cornejo · Patricio Henriquez

---

## El problema

Dadas `M` ubicaciones candidatas y `N` clientes, elegir el subconjunto `S` de instalaciones a
abrir que minimice el costo total:

```
Z(S) = Σ f_i  +  Σ min c_ij
      i∈S      j∈C  i∈S
```

El primer término son los **costos fijos** de abrir cada instalación; el segundo, el costo de
**conectar** cada cliente con la instalación abierta más cercana. Los dos están en tensión:
abrir más instalaciones encarece los costos fijos pero abarata la conexión.

Es NP-Hard, con un espacio de búsqueda de `2^M` subconjuntos. Pero como no hay restricciones
de capacidad, **fijado `S` la asignación de clientes es trivial**: cada uno va a la abierta más
barata. Por eso el subconjunto de abiertas describe por completo una solución, y es el objeto
que los modelos aprenden a construir.

---

## Resultados

Gap respecto del experto (ILS), promedio sobre 25 instancias nuevas de `M = 30`, `N = 200`.
Menor es mejor.

| Método | Z medio | Gap vs ILS | Accuracy | Tiempo |
|---|---:|---:|---:|---:|
| Greedy Add *(heurística clásica)* | 40,97 | 4,20 % | — | 0,001 s |
| MLP aplanada, sin ajustar | 41,00 | 4,24 % | 0,660 | 0,005 s |
| MLP aplanada, ajustada | 39,77 | 1,15 % | 0,776 | 0,005 s |
| Scorer equivariante | 39,44 | **0,30 %** | 0,921 | 0,009 s |
| Modelo + Hill Climbing | 39,33 | 0,03 % | — | 0,004 s |
| ILS *(experto)* | 39,32 | 0,00 % | — | 0,139 s |

Dos lecturas que vale la pena destacar:

**La MLP recién salida de la caja no le gana a Greedy Add.** Con ~350.000 parámetros para
14.000 ejemplos, memoriza en vez de aprender la regla de decisión. El ajuste de
hiperparámetros —que encontró que *menos capacidad generaliza mejor*— lleva el gap de 4,24 % a
1,15 %.

**La arquitectura equivariante transfiere a tamaños que nunca vio.** Entrenada solo con
`M = 30`, evaluada sin reentrenar un peso:

| Tamaño | Greedy Add | Modelo | Modelo + HC |
|---|---:|---:|---:|
| `M = 30, N = 200` *(entrenamiento)* | 4,20 % | 0,30 % | 0,06 % |
| `M = 60, N = 400` *(zero-shot)* | 5,03 % | 1,06 % | 0,00 % |
| `M = 100, N = 800` *(zero-shot)* | 4,26 % | 1,05 % | −0,01 % |

Una MLP aplanada ni siquiera puede ejecutarse en esas instancias: su capa de salida tiene
exactamente 31 neuronas.

---

## Estructura

```
.
├── uflp/
│   ├── instance.py        UFLP_Instance · generación de instancias euclidianas
│   ├── state.py           UFLP_State · cachés best_cost / second_cost
│   ├── environment.py     UFLP_Environment · acciones, deltas O(N), transiciones
│   ├── agents.py          GreedyAgent · LocalSearchAgent · ILS · SingleAgentSolver
│   ├── features.py        state2vec
│   ├── data.py            descomposición de soluciones · generate_imitation_data
│   └── models.py          UFLP_MLP · UFLP_Equivariante · UFLP_Attention
├── notebooks/
│   ├── 01_imitacion_mlp.ipynb       MLP aplanada + ajuste de hiperparámetros
│   ├── 02_atencion_puntero.ipynb    encoder de atención + puntero con máscara
│   └── 03_transformer_am.ipynb      encoder-decoder completo (en curso)
```

---

## El ambiente

Las tres clases siguen la arquitectura modular del curso, con separación estricta entre los
datos invariantes, la representación de una solución y las reglas del mundo.

**`UFLP_Instance`** — costos fijos, matriz de costos de conexión, coordenadas. Solo lectura
durante toda la búsqueda.

**`UFLP_State`** — el conjunto `S` de abiertas más tres cachés que abaratan la evaluación de
movimientos: `assign`, `best_cost` y `second_cost`. El segundo es el **truco de Whitaker
(1983)**, base de las implementaciones modernas de búsqueda local para UFLP: al cerrar una
instalación, cada cliente suyo se va exactamente a su segunda mejor opción, sin recorrer `S`.

```
ΔZ_drop(i₁) = −f_i₁ + Σ ( second_cost[j] − best_cost[j] )
                    j: assign[j] = i₁
```

**`UFLP_Environment`** — genera acciones bajo demanda, evalúa deltas en `O(N)` y aplica
transiciones. El vecindario unificado `Add ∪ Drop ∪ Swap` es el que da la cota de aproximación
de factor 3 de Arya et al.

---

## Los algoritmos tradicionales

| Algoritmo | Qué hace |
|---|---|
| **Greedy Add** | Abre en cada paso la instalación de mayor ahorro neto, hasta que ninguna mejore |
| **Hill Climbing** | Búsqueda local sobre `Add ∪ Drop ∪ Swap`, hasta un óptimo local |
| **ILS** | Perturba con `k = 2` swaps aleatorios, reoptimiza, acepta si mejora |

Benchmark del módulo 1 sobre cinco instancias sintéticas:

| Instancia | `M`, `N` | Greedy Add | Hill Climbing | ILS |
|---|---|---:|---:|---:|
| Pequeña | 25, 100 | 3.165,08 | 3.095,78 | **3.092,75** |
| Mediana | 75, 500 | 10.249,45 | 10.249,45 | 10.249,45 |
| Grande | 150, 1.500 | 25.125,70 | 24.446,92 | **24.261,14** |
| Muy grande | 250, 3.500 | 51.487,45 | 49.964,57 | **49.350,18** |
| Mega escala | 400, 6.000 | 79.857,20 | **76.345,56** | 76.345,56 |

El experto que imitan los modelos es el pipeline completo: **Greedy Add → Hill Climbing → ILS**.
Imitar solo a Greedy Add no tendría sentido —es determinista y barato, la red en el mejor caso
empataría—; imitando al ILS la red aprende a construir en una sola pasada soluciones que el
constructivo voraz no alcanza.

---

## Generación de datos de imitación

El experto devuelve un **conjunto** `S*`, no una secuencia, y la imitación paso a paso necesita
un orden. Usamos el **orden voraz restringido a `S*`**: en cada paso se abre, entre las
instalaciones de `S*` que faltan, la de mayor ahorro inmediato. Es determinista, así que
estados parecidos nunca reciben etiquetas contradictorias.

```
S = { }        →  Y = abrir 14
S = {14}       →  Y = abrir 3
S = {14, 3}    →  Y = abrir 20
    ⋮
S = S*         →  Y = STOP
```

Un episodio con `|S*| = k` produce `k + 1` muestras. La última es la decisión de **detenerse**,
que en el UFLP no es trivial: el tamaño de la solución es parte de lo que hay que aprender.

---

## Los tres modelos

### 1 · MLP sobre el estado aplanado

La entrada `M × 12` se aplana a un vector de 360 valores y una capa densa produce `M + 1`
logits. Funciona, pero arrastra dos problemas: **exige `M` fijo**, y la instalación `i` y el
logit `i` están conectados por pesos distintos a los de la instalación `j`, aunque la pregunta
"¿conviene abrir esta?" sea idéntica.

El ajuste de hiperparámetros exploró capas, neuronas, activación, batch y dropout, más *data
augmentation* por permutación de índices. Hallazgo principal: con 14.000 ejemplos, **menos
capacidad generaliza mejor** — ganó una sola capa oculta de 128 neuronas.

### 2 · Scorer equivariante (DeepSets)

Encoder compartido por fila, pooling global `mean ⊕ max`, scorer compartido. Ningún peso
depende de `M`, así que el mismo modelo acepta instancias de cualquier tamaño. Es el que mejor
rinde y el que transfiere zero-shot.

### 3 · Encoder-decoder con puntero *(en curso)*

La arquitectura **Glimpse** de Kool et al. adaptada al UFLP: encoder de 3 capas de
self-attention, contexto de decisión, capa de glimpse y puntero con máscara.

```
ENCODER                               DECODER
X (M+1, 9)                            contexto = W · [h_graph ; h_última ; h_stop]
  → Linear 9 → 128                      → GLIMPSE  MHA(q=contexto, K=V=H) → q′
  → ×3 (MHA + skip + LN                 → PUNTERO  u_i = 10·tanh(q′ᵀk_i/√d)
        FF + skip + LN)                 → máscara  u_i = −1e9 si no válida
  → H (M+1, 128)                        → softmax → p ∈ R^(M+1)
    h_graph = mean(H)
```

744.064 parámetros entrenables, ninguno dependiente de `M`. El 80 % está en el encoder.

**La fila STOP.** Un puntero solo sabe apuntar a filas, y "detenerse" no corresponde a ninguna
instalación. La solución es agregar **una fila extra** a la matriz: un pseudo-elemento que
representa la acción de parar y transporta la información global del estado. Así STOP compite
en la misma softmax que las aperturas, sin necesidad de una cabeza de salida aparte, y la
independencia del tamaño queda intacta.

## Arquitectura del Transformer
<img width="789" height="847" alt="image" src="https://github.com/user-attachments/assets/eaccfebb-e5f3-4ecc-9759-eb6b7d2904bc" />


---

## Dos diferencias con el paper original

**LayerNorm en vez de BatchNorm.** Kool et al. usan BatchNorm, que calcula estadísticas sobre
el lote. Nosotros evaluamos instancia por instancia y en tamaños que no vimos entrenando, donde
las estadísticas acumuladas quedarían mal calibradas. LayerNorm no depende del lote ni del
número de filas.

**El encoder corre en cada paso.** En el TSP las features de cada ciudad son sus coordenadas,
estáticas, así que `H` se calcula una vez y se reutiliza. En el UFLP seis de las nueve columnas
dependen del estado —cuánto ahorra abrir una instalación depende de cuáles ya están abiertas—
así que hay que volver a codificar en cada decisión. A esta escala el costo es despreciable:
un rollout completo toma ~14 ms contra ~140 ms del ILS.

---

## Uso

```bash
git clone https://github.com/<usuario>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

```python
from uflp import random_instance, UFLP_State, greedy_add, solve_expert

inst = random_instance(M=30, N=200, rng=np.random.default_rng(0))

g = greedy_add(inst)          # heurística clásica
e = solve_expert(inst)        # Greedy Add → Hill Climbing → ILS

print(g)                      # UFLP_State(|S|=11, Z=35.332)
print(e)                      # UFLP_State(|S|=10, Z=33.779)
```

Los notebooks corren de principio a fin en Colab sin instalar nada fuera de lo que ya trae.
Cada uno expone una constante `FAST` al inicio para reducir el dataset y correr en pocos
minutos mientras se itera.

---

## Estado

- [x] Ambiente UFLP, agentes y benchmark de algoritmos tradicionales
- [x] Generación de datos de imitación y descomposición de soluciones
- [x] MLP aplanada + ajuste de hiperparámetros
- [x] Scorer equivariante y transferencia entre tamaños
- [x] Diseño del encoder-decoder con puntero y validación sin entrenar
- [ ] Entrenamiento del transformer
- [ ] Integración en el agente constructivo y rollout
- [ ] Gap contra el óptimo exacto (MIP)
- [ ] Evaluación por tamaño de instancia

---

## Una advertencia honesta

El UFLP es NP-Hard, pero **empíricamente fácil de resolver a optimalidad**: la relajación LP de
la formulación desagregada es muy ajustada, así que un solver MIP encuentra el óptimo de estas
instancias en segundos. Los modelos no compiten con eso, ni pretenden hacerlo.

Lo que sí muestran los resultados es dónde está el aporte real de un modelo aprendido: como
**constructor rápido**, que produce en una sola pasada un punto de partida mucho mejor que una
heurística voraz, y sobre el cual una búsqueda local converge en menos iteraciones.

---

## Referencias

- Kool, W., van Hoof, H. & Welling, M. (2019). *Attention, Learn to Solve Routing Problems!* ICLR.
- Whitaker, R. (1983). *A fast algorithm for the greedy interchange for large-scale clustering and median location problems.* INFOR 21(2), 95-108.
- Resende, M. & Werneck, R. (2006). *A hybrid multistart heuristic for the uncapacitated facility location problem.* EJOR 174(1), 54-68.
- Arya, V. et al. (2001). *Local search heuristics for k-median and facility location problems.* STOC '01, 21-29.
- Cornuejols, G., Fisher, M. & Nemhauser, G. (1977). *On the uncapacitated location problem.* Annals of Discrete Mathematics 1, 163-177.
- Zaheer, M. et al. (2017). *Deep Sets.* NeurIPS 30.
- Ross, S., Gordon, G. & Bagnell, D. (2011). *A reduction of imitation learning and structured prediction to no-regret online learning* (DAgger). AISTATS.
- Vinyals, O., Fortunato, M. & Jaitly, N. (2015). *Pointer Networks.* NeurIPS 28.


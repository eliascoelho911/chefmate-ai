# Chefmate AI — Domain Glossary

> Vocabulário ubíquo do projeto.  Novos termos são adicionados à medida
> que o domínio evolui; nunca se assume que o leitor já conhece um
> conceito que não esteja aqui definido.

---

## Intent

Objetivo do usuário expresso em linguagem natural.

A detecção de intent é responsabilidade do módulo `IntentDetector`, que
mapeia uma frase livre para um valor da enumeração `Intent`.  Cada
intent representa *o que* o usuário quer, não *como* o sistema deve
satisfazê-lo.

Valores possíveis:

| Valor | Significado |
|-------|-------------|
| `INGREDIENT_SEARCH` | Encontrar receitas a partir de ingredientes disponíveis. |
| `SPECIFIC_RECIPE` | Buscar uma receita conhecida pelo nome. |
| `RECIPE_GENERATION` | Sugerir receitas de forma genérica (refeição, ocasião). |
| `STEP_NAVIGATION` | Navegar pelos passos de uma receita já selecionada. |
| `DIET_FILTER` | Filtrar por restrições alimentares. |
| `NUTRITION_INFO` | Obter informações nutricionais. |
| `TIME_FILTER` | Filtrar por tempo de preparo. |
| `RATING_FILTER` | Buscar receitas bem avaliadas. |
| `UNCLEAR` | Intenção não identificada. |

**Regra:** módulos downstream (prompt builder, retriever) recebem um
`Intent` tipado; nunca trabalham com strings literais.

---

## Recipe

Receita culinária normalizada.

Representada pelo modelo `Recipe` (Pydantic).  Contém:
`faiss_index`, `name`, `ingredients_with_quantities`,
`recipe_instructions`, `category`, `calories`, `total_time`, `rating`,
`images`.

**Regra:** o `RecipeRepository` é o único módulo que mapeia linhas do
SQLite para instâncias de `Recipe`.  Nenhum outro módulo conhece o
schema do banco.

---

## VectorIndex

Adapter que esconde todos os detalhes do FAISS.

Interface: `search(embedding, index_name, top_k) → (distances, indices)`.

**Regra:** nenhum módulo fora de `VectorIndex` importa `faiss`
diretamente.

---

## RecipeRepository

Adapter que satisfaz o seam de persistência de receitas usando SQLite.

Interface: `get_by_indices(indices) → List[Recipe]`.

**Regra:** batch-first — sempre recebe uma lista de índices e retorna
`Recipe` tipados, filtrando misses internamente.

---

## RecipeRetriever

Módulo profundo que orquestra busca vetorial + hidratação de metadados.

Interface: `retrieve(query_embedding, intent, top_k) → List[Recipe]`.

**Regra:** conhece o mapeamento `Intent → nome_do_índice_FAISS`, mas
não conhece detalhes de como o índice é carregado ou como o SQLite é
acessado.

---

## RecipeSearch

Módulo profundo que orquestra o pipeline completo de busca:
embed → detect intent → retrieve.

Interface pública: `search(query, top_k) → List[Recipe]`.

**Regra:** a detecção de intent é um passo interno.  Callers que já
conhecem a intent (ex.: endpoints da API) podem passá-la via o
parâmetro opcional `intent`, mas nunca precisam chamar o detector
diretamente.

---

## ChatOrchestrator

Módulo profundo que orquestra uma volta conversacional completa:
validate → detect intent → search → build prompt → stream LLM.

**Regra:** é o único módulo que coordena detector + busca + prompt
numa única transação.  Nenhum outro módulo conhece a sequência
completa.

---

## LLMRunner

Adapter que satisfaz o seam de inferência de linguagem via OpenRouter.

Interface: `stream_response(messages) → Iterator[str]`.

**Regra:** recebe `api_key` e `model` por injeção de dependência;
nunca lê configuração global internamente.

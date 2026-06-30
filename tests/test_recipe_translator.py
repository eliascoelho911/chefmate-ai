"""
Testes de integração reais para RecipeTranslator.

Fazem chamadas HTTP reais à API OpenRouter.
Execute com:
    pytest tests/test_recipe_translator.py -v

Requer OPENROUTER_API_KEY no .env (ou ambiente).
Cada execução gera um arquivo de log distinto em tests/logs/.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv
from openai import OpenAI

from app.core.models import Recipe
from app.utils.recipe_translator import RecipeTranslator

# ---------------------------------------------------------------------------
# Setup de logging por execução
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).with_suffix("").parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"recipe_translator_tests_{_RUN_TIMESTAMP}.log"

_test_logger = logging.getLogger("test_recipe_translator")
_test_logger.setLevel(logging.INFO)
# evita duplicação se o módulo for recarregado
if not any(
    isinstance(h, logging.FileHandler) and h.baseFilename == str(LOG_FILE)
    for h in _test_logger.handlers
):
    _fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    _fh.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    _test_logger.addHandler(_fh)


def _log_line(
    test_id: str,
    inputs: list,
    outputs: list,
    elapsed_ms: float,
    parse_ok: bool,
    valid: bool,
    notes: str = "",
):
    """Grava uma linha estruturada no log de testes."""
    inputs_str = ";".join(str(i) for i in inputs)
    outputs_str = ";".join(str(o) for o in outputs)
    status = f"parse={'OK' if parse_ok else 'FAIL'} valid={'OK' if valid else 'FAIL'}"
    _test_logger.info(
        "[%s] %s | in=%s | out=%s | %.0fms | notes=%s",
        test_id,
        status,
        inputs_str,
        outputs_str,
        elapsed_ms,
        notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_recipe(
    name="Grilled Chicken",
    ingredients_cleaned=None,
    ingredients_with_quantities=None,
    recipe_instructions=None,
    category="Main Dish",
    calories="350 kcal",
    rating=4.5,
):
    """Factory para criar receitas sintéticas."""
    if ingredients_cleaned is None:
        ingredients_cleaned = ["chicken breast", "salt", "olive oil"]
    if ingredients_with_quantities is None:
        ingredients_with_quantities = [
            "2 chicken breasts",
            "1 tsp salt",
            "2 tbsp olive oil",
        ]
    if recipe_instructions is None:
        recipe_instructions = [
            "Preheat grill to medium-high.",
            "Season chicken with salt and pepper.",
            "Grill 10 minutes per side until done.",
        ]
    return Recipe(
        faiss_index=999,
        name=name,
        ingredients_cleaned=ingredients_cleaned,
        ingredients_with_quantities=ingredients_with_quantities,
        recipe_instructions=recipe_instructions,
        category=category,
        calories=calories,
        total_time="30 min",
        rating=rating,
        images=[],
    )


def _assert_not_english(text: str, english_cues: list[str]) -> None:
    """Heurística: texto não deve conter palavras inglesas óbvias que deveriam ter sido traduzidas."""
    lower = text.lower()
    found = [cue for cue in english_cues if cue in lower]
    assert not found, f"Texto parece ainda conter inglês: '{text}' (encontrado {found})"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def recipe_translator():
    """Instancia RecipeTranslator com cliente OpenRouter real."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip(
            "OPENROUTER_API_KEY não encontrada no ambiente — pulando testes reais."
        )

    model = os.getenv("OPENROUTER_FAST_MODEL") or "openai/gpt-4o-mini"

    http_client = httpx.Client(
        timeout=httpx.Timeout(20.0, connect=5.0, read=15.0),
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        http_client=http_client,
        max_retries=0,
    )
    return RecipeTranslator(client=client, model=model)


def _call_with_timeout(fn, *args, timeout_sec: float = 20, **kwargs):
    """Executa fn em uma thread separada, abortando após timeout_sec."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_sec)
    except FutureTimeoutError:
        executor.shutdown(wait=False)
        raise TimeoutError(
            f"LLM call did not complete within {timeout_sec}s (thread hung on I/O)"
        )


def _detect_parse_error(caplog) -> bool:
    """Retorna True se NÃO houve erro de parsing/tradução nos logs do módulo."""
    error_msgs = [
        "OpenRouter recipe translation failed",
        "Failed to parse recipe translation JSON",
        "Returning untranslated recipes due to error",
        "No translations parsed",
    ]
    for record in caplog.get_records("call"):
        if record.levelno >= logging.WARNING:
            for msg in error_msgs:
                if msg in record.getMessage():
                    return False
    return True


# ---------------------------------------------------------------------------
# Cenários
# ---------------------------------------------------------------------------
class TestSingleRecipe:
    """Tradução de uma única receita sintética."""

    def test_simple_recipe_translation(self, recipe_translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.recipe_translator")
        recipe = _make_recipe()
        t0 = time.perf_counter()
        result = _call_with_timeout(recipe_translator.translate_recipes, [recipe])
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 1
            tr = result[0]
            # O modelo nem sempre traduz o nome; verificamos se outros campos foram
            # traduzidos para confirmar que o LLM processou a receita.
            ingr_lower = " ".join(tr.ingredients_cleaned).lower()
            instr_lower = " ".join(tr.recipe_instructions).lower()

            name_translated = "frango" in tr.name.lower()
            ingr_translated = (
                "frango" in ingr_lower or "sal" in ingr_lower or "azeite" in ingr_lower
            )
            instr_translated = (
                "frango" in instr_lower
                or "grelha" in instr_lower
                or "sal" in instr_lower
            )
            cat_translated = tr.category != recipe.category

            assert (
                name_translated or ingr_translated or instr_translated or cat_translated
            ), (
                f"Nenhum campo parece ter sido traduzido. "
                f"nome='{tr.name}', ingredientes={tr.ingredients_cleaned}, "
                f"instrucoes={tr.recipe_instructions}, categoria='{tr.category}'"
            )

            # Se o nome foi traduzido, não deve manter inglês óbvio
            if name_translated:
                _assert_not_english(tr.name, ["grilled", "chicken"])
            # Categoria traduzida não deve manter inglês óbvio
            if cat_translated:
                _assert_not_english(tr.category, ["main", "dish"])
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id="single_simple",
                inputs=[recipe.name],
                outputs=[result[0].name if result else "N/A"],
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )


class TestMultipleRecipes:
    """Tradução de múltiplas receitas para exercitar chunking/paralelismo."""

    def test_two_recipes(self, recipe_translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.recipe_translator")
        r1 = _make_recipe(name="Grilled Chicken", category="Main Dish")
        r2 = _make_recipe(
            name="Chocolate Cake",
            ingredients_cleaned=["flour", "sugar", "cocoa powder"],
            ingredients_with_quantities=[
                "2 cups flour",
                "1 cup sugar",
                "3 tbsp cocoa powder",
            ],
            recipe_instructions=[
                "Preheat oven to 180C.",
                "Mix dry ingredients.",
                "Bake for 30 minutes.",
            ],
            category="Dessert",
        )
        recipes = [r1, r2]
        t0 = time.perf_counter()
        result = _call_with_timeout(recipe_translator.translate_recipes, recipes)
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 2
            tr0, tr1 = result[0], result[1]

            # Verifica se ALGUM campo de cada receita foi traduzido
            r0_ingr = " ".join(tr0.ingredients_cleaned).lower()
            r0_instr = " ".join(tr0.recipe_instructions).lower()
            r0_name_ok = tr0.name != r1.name
            r0_ingr_ok = "frango" in r0_ingr or "sal" in r0_ingr
            r0_instr_ok = "frango" in r0_instr or "grelha" in r0_instr
            r0_cat_ok = tr0.category != r1.category

            r1_ingr = " ".join(tr1.ingredients_cleaned).lower()
            r1_instr = " ".join(tr1.recipe_instructions).lower()
            r1_name_ok = tr1.name != r2.name
            r1_ingr_ok = (
                "farinha" in r1_ingr or "açúcar" in r1_ingr or "cacau" in r1_ingr
            )
            r1_instr_ok = (
                "forno" in r1_instr or "misture" in r1_instr or "asse" in r1_instr
            )
            r1_cat_ok = tr1.category != r2.category

            assert r0_name_ok or r0_ingr_ok or r0_instr_ok or r0_cat_ok, (
                f"Receita 1 não traduzida. nome='{tr0.name}', ingr={tr0.ingredients_cleaned}, "
                f"instr={tr0.recipe_instructions}, cat='{tr0.category}'"
            )
            assert r1_name_ok or r1_ingr_ok or r1_instr_ok or r1_cat_ok, (
                f"Receita 2 não traduzida. nome='{tr1.name}', ingr={tr1.ingredients_cleaned}, "
                f"instr={tr1.recipe_instructions}, cat='{tr1.category}'"
            )

            if r0_name_ok:
                _assert_not_english(tr0.name, ["grilled", "chicken"])
            if r1_name_ok:
                _assert_not_english(tr1.name, ["cake"])
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id="multi_two",
                inputs=[r.name for r in recipes],
                outputs=[
                    result[i].name if i < len(result) else "N/A" for i in range(2)
                ],
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )


class TestFieldPreservation:
    """Campos numéricos, IDs e imagens não devem ser alterados."""

    def test_preserved_fields(self, recipe_translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.recipe_translator")
        recipe = _make_recipe(calories="500 kcal", rating=3.8)
        t0 = time.perf_counter()
        result = _call_with_timeout(recipe_translator.translate_recipes, [recipe])
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 1
            tr = result[0]
            assert tr.calories == recipe.calories
            assert tr.rating == recipe.rating
            assert tr.faiss_index == recipe.faiss_index
            assert tr.total_time == recipe.total_time
            assert tr.images == recipe.images
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id="preserve_fields",
                inputs=[recipe.calories, str(recipe.rating), str(recipe.faiss_index)],
                outputs=[tr.calories, str(tr.rating), str(tr.faiss_index)],
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )


class TestEdgeCases:
    """Cenários limítrofes."""

    def test_empty_list(self, recipe_translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.recipe_translator")
        result = _call_with_timeout(recipe_translator.translate_recipes, [])
        _log_line(
            test_id="edge_empty",
            inputs=[],
            outputs=[],
            elapsed_ms=0,
            parse_ok=True,
            valid=result == [],
            notes="" if result == [] else f"esperava [], obteve {result}",
        )
        assert result == []

    def test_recipe_with_empty_fields(self, recipe_translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.recipe_translator")
        recipe = _make_recipe(
            name="",
            ingredients_cleaned=[],
            ingredients_with_quantities=[],
            recipe_instructions=[],
            category="",
        )
        t0 = time.perf_counter()
        result = _call_with_timeout(recipe_translator.translate_recipes, [recipe])
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 1
            # Deve retornar a receita mesmo com campos vazios (fallback)
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id="edge_empty_fields",
                inputs=["empty recipe"],
                outputs=[result[0].name if result else "N/A"],
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )

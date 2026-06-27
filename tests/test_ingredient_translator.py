"""
Testes de integração reais para IngredientTranslator.

Fazem chamadas HTTP reais à API OpenRouter.
Execute com:
    pytest tests/test_ingredient_translator.py -v

Requer OPENROUTER_API_KEY no .env (ou ambiente).
Cada execução gera um arquivo de log distinto em tests/logs/.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import OpenAI

from app.utils.ingredient_translator import IngredientTranslator

# ---------------------------------------------------------------------------
# Setup de logging por execução
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).with_suffix("").parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"ingredient_translator_tests_{_RUN_TIMESTAMP}.log"

_test_logger = logging.getLogger("test_ingredient_translator")
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
    inputs_str = ";".join(inputs)
    outputs_str = ";".join(outputs)
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
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def translator():
    """Instancia IngredientTranslator com cliente OpenRouter real."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip(
            "OPENROUTER_API_KEY não encontrada no ambiente — pulando testes reais."
        )

    fast_model = os.getenv("OPENROUTER_FAST_MODEL") or os.getenv("OPENROUTER_MODEL")
    if not fast_model:
        fast_model = "meta-llama/llama-3.1-8b-instruct"

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return IngredientTranslator(client=client, model=fast_model)


def _detect_parse_error(caplog) -> bool:
    """Retorna True se NÃO houve erro de parsing/tradução nos logs do módulo."""
    error_msgs = [
        "OpenRouter translation failed",
        "Failed to parse translation JSON",
        "Returning untranslated ingredients due to error",
        "No translations parsed",
        "Missing translation for",
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
class TestSimpleIngredients:
    """Traduções diretas sem necessidade de normalização."""

    @pytest.mark.parametrize(
        "ingredient,expected_substring",
        [
            ("frango", "chicken"),
            ("arroz", "rice"),
            ("brocolis", "broccoli"),
            ("ovo", "egg"),
            ("leite", "milk"),
            ("batata", "potato"),
            ("cenoura", "carrot"),
            ("tomate", "tomato"),
            ("cebola", "onion"),
            ("alho", "garlic"),
        ],
    )
    def test_simple_translation(
        self, translator, ingredient, expected_substring, caplog
    ):
        caplog.set_level(logging.DEBUG, logger="app.utils.ingredient_translator")
        t0 = time.perf_counter()
        result = translator.translate_batch([ingredient])
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 1
            translated = result[0].lower()
            assert expected_substring in translated, (
                f"Esperava que '{ingredient}' contivesse '{expected_substring}', mas obteve '{translated}'"
            )
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id=f"simple_{ingredient}",
                inputs=[ingredient],
                outputs=result,
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )


class TestNormalization:
    """
    Ingredientes específicos que o prompt pede para generalizar.
    O LLM deve preferir termos amplos (ex: 'chicken' ao invés de 'chicken breast').
    """

    @pytest.mark.parametrize(
        "ingredient,expected_substring,forbidden",
        [
            ("peito de frango", "chicken", ["breast"]),
            ("arroz integral", "rice", ["brown"]),
            ("batata doce", "sweet potato", []),
            ("filé mignon", "beef", []),
            ("azeite de oliva", "olive oil", []),
            ("farinha de trigo", "flour", ["wheat"]),
            ("açúcar mascavo", "sugar", ["brown"]),
            ("creme de leite", "cream", []),
        ],
    )
    def test_generic_translation(
        self, translator, ingredient, expected_substring, forbidden, caplog
    ):
        caplog.set_level(logging.DEBUG, logger="app.utils.ingredient_translator")
        t0 = time.perf_counter()
        result = translator.translate_batch([ingredient])
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 1
            translated = result[0].lower()

            assert expected_substring in translated, (
                f"Esperava que '{ingredient}' contivesse '{expected_substring}', mas obteve '{translated}'"
            )
            for f in forbidden:
                assert f not in translated, (
                    f"'{ingredient}' não deveria conter '{f}', mas obteve '{translated}'"
                )
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id=f"norm_{ingredient}",
                inputs=[ingredient],
                outputs=result,
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )


class TestBatchBehavior:
    """Comportamentos do batch: deduplicação, preservação de ordem, termos em inglês."""

    def test_duplicate_preserves_order(self, translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.ingredient_translator")
        ingredients = ["frango", "frango", "arroz"]
        t0 = time.perf_counter()
        result = translator.translate_batch(ingredients)
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 3
            assert result[0] == result[1], "Duplicatas devem ter a mesma tradução"
            assert result[0] != result[2], (
                "Itens diferentes devem ter traduções distintas"
            )
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id="batch_dupes",
                inputs=ingredients,
                outputs=result,
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )

    def test_already_english(self, translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.ingredient_translator")
        ingredients = ["chicken", "rice", "broccoli"]
        t0 = time.perf_counter()
        result = translator.translate_batch(ingredients)
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 3
            for original, translated in zip(ingredients, result):
                assert original == translated, (
                    f"Termo já em inglês '{original}' foi alterado para '{translated}'"
                )
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id="batch_english",
                inputs=ingredients,
                outputs=result,
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )

    def test_mixed_batch(self, translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.ingredient_translator")
        ingredients = ["frango", "arroz integral", "broccoli"]
        t0 = time.perf_counter()
        result = translator.translate_batch(ingredients)
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 3
            assert "chicken" in result[0]
            assert "rice" in result[1]
            assert "broccoli" == result[2]
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id="batch_mixed",
                inputs=ingredients,
                outputs=result,
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )


class TestLiteralTranslation:
    """
    Quando generalize=False, o LLM deve preservar termos específicos.
    """

    @pytest.mark.parametrize(
        "ingredient,expected_substring",
        [
            ("peito de frango", "breast"),
            ("arroz integral", "brown"),
            ("batata doce", "sweet potato"),
            ("filé mignon", "tenderloin"),
            ("azeite de oliva", "olive oil"),
            ("farinha de trigo", "wheat"),
            ("açúcar mascavo", "brown"),
            ("creme de leite", "cream"),
        ],
    )
    def test_specific_translation(
        self, translator, ingredient, expected_substring, caplog
    ):
        caplog.set_level(logging.DEBUG, logger="app.utils.ingredient_translator")
        t0 = time.perf_counter()
        result = translator.translate_batch([ingredient], generalize=False)
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 1
            translated = result[0].lower()
            assert expected_substring in translated, (
                f"Esperava que '{ingredient}' contivesse '{expected_substring}', mas obteve '{translated}'"
            )
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id=f"literal_{ingredient}",
                inputs=[ingredient],
                outputs=result,
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )

    def test_literal_mixed_batch(self, translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.ingredient_translator")
        ingredients = ["peito de frango", "arroz integral", "broccoli"]
        t0 = time.perf_counter()
        result = translator.translate_batch(ingredients, generalize=False)
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 3
            assert "breast" in result[0].lower()
            assert "brown" in result[1].lower()
            assert result[2] == "broccoli"
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id="literal_mixed",
                inputs=ingredients,
                outputs=result,
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )


class TestEdgeCases:
    """Cenários limítrofes."""

    def test_empty_list(self, translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.ingredient_translator")
        result = translator.translate_batch([])
        _log_line(
            test_id="edge_empty",
            inputs=[],
            outputs=result,
            elapsed_ms=0,
            parse_ok=True,
            valid=result == [],
            notes="" if result == [] else f"esperava [], obteve {result}",
        )
        assert result == []

    def test_whitespace_only(self, translator, caplog):
        caplog.set_level(logging.DEBUG, logger="app.utils.ingredient_translator")
        ingredients = ["   ", "frango", ""]
        t0 = time.perf_counter()
        result = translator.translate_batch(ingredients)
        elapsed = (time.perf_counter() - t0) * 1000

        parse_ok = _detect_parse_error(caplog)
        valid = False
        notes = ""

        try:
            assert len(result) == 3
            assert result[1] == "chicken" or "chicken" in result[1].lower()
            valid = True
        except AssertionError as exc:
            notes = str(exc)
            raise
        finally:
            _log_line(
                test_id="edge_whitespace",
                inputs=ingredients,
                outputs=result,
                elapsed_ms=elapsed,
                parse_ok=parse_ok,
                valid=valid,
                notes=notes,
            )

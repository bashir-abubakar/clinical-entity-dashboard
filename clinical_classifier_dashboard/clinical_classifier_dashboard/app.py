from __future__ import annotations

import html
import hashlib
import json
import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import declare_component
import torch
import torch.nn.functional as F
from setfit import SetFitModel
from torch import nn
from transformers import AutoModel, AutoTokenizer


entity_selector = declare_component(
    "clinical_entity_selector",
    path=str(Path(__file__).resolve().parent / "entity_selector"),
)


def prepare_selected_input(text, selection, revision, section, context_words):
    if not isinstance(selection, dict) or selection.get("revision") != revision:
        return None
    if selection.get("clear"):
        return None
    start, end = selection.get("start"), selection.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("Highlight a valid entity in the current text.")
    entity = text[start:end]
    if not entity.strip() or entity != selection.get("text"):
        raise ValueError("The selection no longer matches the text. Highlight it again.")
    occurrence = 1 + sum(a < start for a, b in find_occurrences(text, entity))
    return PreparedInput(
        clinical_text=text, entity_text=entity,
        section=normalise_section(section), occurrence=occurrence,
        entity_start=start, entity_end=end,
        training_text=make_training_text(text, start, end, section, context_words),
    )


ENTITY_TYPES = ["DIAGNOSIS", "MEDICATION", "SYMPTOM"]
CONTEXT_LABELS = ["HISTORICAL", "NEGATED", "PRESENT", "UNCERTAIN"]
ENTITY_START = "[ENTITY]"
ENTITY_END = "[/ENTITY]"
CLASS_COLOURS = {
    "DIAGNOSIS": "#8b7cff",
    "MEDICATION": "#31d7c6",
    "SYMPTOM": "#ffb86b",
    "HISTORICAL": "#8b7cff",
    "NEGATED": "#ff6685",
    "PRESENT": "#31d7c6",
    "UNCERTAIN": "#ffb86b",
}


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    setfit_model_dir: Path
    protonet_model_dir: Path
    setfit_development_csv: Path | None
    protonet_development_csv: Path | None
    device: str
    context_words: int
    max_length: int
    top_k_examples: int
    max_explanation_words: int


@dataclass
class SetFitBundle:
    model: SetFitModel
    labels: list[str]
    development: pd.DataFrame | None
    development_embeddings: np.ndarray | None


@dataclass
class ProtoNetBundle:
    tokenizer: Any
    model: "MentionEncoder"
    prototypes: torch.Tensor
    labels: list[str]
    temperature: float
    start_id: int
    end_id: int
    max_length: int
    device: torch.device
    development: pd.DataFrame | None
    development_embeddings: torch.Tensor | None


@dataclass(frozen=True)
class PreparedInput:
    clinical_text: str
    entity_text: str
    section: str
    occurrence: int
    entity_start: int
    entity_end: int
    training_text: str


class MentionEncoder(nn.Module):
    def __init__(self, model_id: str, projection_dim: int) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        hidden_size = int(self.backbone.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.Tanh(),
            nn.LayerNorm(projection_dim),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        start_token_id: int,
        end_token_id: int,
    ) -> torch.Tensor:
        hidden = self.backbone(**batch).last_hidden_state
        pooled: list[torch.Tensor] = []
        for row, input_ids in enumerate(batch["input_ids"]):
            starts = torch.nonzero(
                input_ids.eq(start_token_id), as_tuple=False
            ).flatten()
            ends = torch.nonzero(
                input_ids.eq(end_token_id), as_tuple=False
            ).flatten()
            if len(starts) != 1 or len(ends) != 1:
                raise RuntimeError(
                    "Each input must contain exactly one [ENTITY] and [/ENTITY] marker."
                )
            if int(ends[0]) <= int(starts[0]) + 1:
                raise RuntimeError("The marked entity span is empty.")
            pooled.append(
                hidden[row, int(starts[0]) + 1 : int(ends[0])].mean(dim=0)
            )
        projected = self.projection(torch.stack(pooled))
        return F.normalize(projected.float(), p=2, dim=-1)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #070b14;
            --panel: rgba(17, 25, 40, 0.62);
            --panel-strong: rgba(20, 29, 48, 0.82);
            --border: rgba(255, 255, 255, 0.12);
            --text: #f4f7ff;
            --muted: #a8b0c3;
            --violet: #8b7cff;
            --teal: #31d7c6;
            --amber: #ffb86b;
            --coral: #ff6685;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% 7%, rgba(139,124,255,.20), transparent 28%),
                radial-gradient(circle at 88% 12%, rgba(49,215,198,.14), transparent 24%),
                radial-gradient(circle at 55% 88%, rgba(52,88,180,.12), transparent 30%),
                var(--bg);
            color: var(--text);
        }
        [data-testid="stHeader"] { background: rgba(7,11,20,.35); }
        [data-testid="stSidebar"] {
            background: rgba(9, 14, 25, .86);
            border-right: 1px solid var(--border);
        }
        .block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 4rem; }
        h1, h2, h3, h4, p, label, .stMarkdown { color: var(--text); }
        .hero {
            padding: 1.45rem 1.6rem;
            border: 1px solid var(--border);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(139,124,255,.15), rgba(49,215,198,.08));
            box-shadow: 0 24px 70px rgba(0,0,0,.28);
            backdrop-filter: blur(22px);
            margin-bottom: 1.25rem;
        }
        .hero-kicker { color: var(--teal); font-weight: 700; letter-spacing: .13em; font-size: .74rem; }
        .hero h1 { margin: .35rem 0 .45rem; font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.05; }
        .hero p { color: var(--muted); max-width: 850px; margin: 0; }
        .glass-card {
            min-height: 146px;
            padding: 1.15rem 1.2rem;
            border: 1px solid var(--border);
            border-radius: 21px;
            background: var(--panel);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 18px 45px rgba(0,0,0,.20);
            backdrop-filter: blur(22px);
        }
        .glass-card .label { color: var(--muted); font-size: .77rem; letter-spacing: .08em; text-transform: uppercase; }
        .glass-card .value { font-size: 1.65rem; font-weight: 800; margin-top: .35rem; }
        .glass-card .detail { color: var(--muted); margin-top: .35rem; font-size: .88rem; }
        .accent-violet { box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 18px 50px rgba(139,124,255,.10); }
        .accent-teal { box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 18px 50px rgba(49,215,198,.08); }
        .entity-chip {
            display: inline-block; padding: .18rem .48rem; margin: 0 .1rem;
            border: 1px solid rgba(49,215,198,.45); border-radius: 8px;
            background: rgba(49,215,198,.12); color: #c9fff8; font-weight: 700;
        }
        .input-preview {
            padding: 1rem 1.1rem; border-radius: 16px; border: 1px solid var(--border);
            background: rgba(5,9,17,.62); color: #dfe6f5; line-height: 1.8;
        }
        .explanation-note {
            padding: .85rem 1rem; border-left: 3px solid var(--violet);
            background: rgba(139,124,255,.08); border-radius: 0 14px 14px 0;
            color: #d9def0;
        }
        .review-low { color: var(--teal); }
        .review-moderate { color: var(--amber); }
        .review-high { color: var(--coral); }
        [data-testid="stMetric"] {
            background: var(--panel); border: 1px solid var(--border); border-radius: 18px;
            padding: .8rem 1rem; backdrop-filter: blur(18px);
        }
        [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input,
        [data-baseweb="select"] > div {
            background: rgba(10,16,28,.74) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
            border-radius: 14px !important;
        }
        .stButton > button, .stDownloadButton > button {
            border: 1px solid rgba(255,255,255,.16); border-radius: 14px;
            background: linear-gradient(135deg, rgba(139,124,255,.92), rgba(80,103,226,.92));
            color: white; font-weight: 750; min-height: 44px;
            box-shadow: 0 12px 30px rgba(91,79,210,.22);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(255,255,255,.36); transform: translateY(-1px);
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .5rem; }
        [data-testid="stTabs"] button { border-radius: 12px; }
        .small-muted { color: var(--muted); font-size: .82rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value).strip())))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config() -> AppConfig:
    default_path = Path(__file__).with_name("dashboard_config.toml")
    configured = os.getenv("CLINICAL_DASHBOARD_CONFIG")
    config_path = Path(configured).expanduser().resolve() if configured else default_path
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. Copy "
            "dashboard_config.example.toml to dashboard_config.toml and update the paths."
        )
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    paths = raw.get("paths", {})
    settings = raw.get("app", {})
    config_dir = config_path.parent
    project_root = resolve_path(paths.get("project_root", ".."), config_dir)
    if project_root is None:
        raise ValueError("paths.project_root must be provided.")
    setfit_dir = resolve_path(paths.get("setfit_entity_type_dir"), project_root)
    protonet_dir = resolve_path(paths.get("protonet_context_dir"), project_root)
    if setfit_dir is None or protonet_dir is None:
        raise ValueError("Both model directory paths must be configured.")
    return AppConfig(
        project_root=project_root,
        setfit_model_dir=setfit_dir,
        protonet_model_dir=protonet_dir,
        setfit_development_csv=resolve_path(
            paths.get("setfit_development_csv"), project_root
        ),
        protonet_development_csv=resolve_path(
            paths.get("protonet_development_csv"), project_root
        ),
        device=str(settings.get("device", "auto")).lower(),
        context_words=int(settings.get("context_words", 15)),
        max_length=int(settings.get("max_length", 256)),
        top_k_examples=int(settings.get("top_k_examples", 3)),
        max_explanation_words=int(settings.get("max_explanation_words", 40)),
    )


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("app.device must be auto, cpu or cuda.")
    return torch.device(requested)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def normalise_section(value: str) -> str:
    section = " ".join(str(value).strip().split())
    return section or "unknown"


def find_occurrences(text: str, entity: str) -> list[tuple[int, int]]:
    if not text.strip() or not entity.strip():
        return []
    return [
        (match.start(), match.end())
        for match in re.finditer(re.escape(entity.strip()), text, flags=re.IGNORECASE)
    ]


def make_training_text(
    clinical_text: str,
    start: int,
    end: int,
    section: str,
    context_words: int,
) -> str:
    left_tokens = re.findall(r"\S+", clinical_text[:start])
    right_tokens = re.findall(r"\S+", clinical_text[end:])
    left = " ".join(left_tokens[-context_words:])
    right = " ".join(right_tokens[:context_words])
    components = [f"SECTION: {normalise_section(section)}.", "CONTEXT:"]
    if left:
        components.append(left)
    entity = " ".join(clinical_text[start:end].strip().split())
    components.append(f"{ENTITY_START} {entity} {ENTITY_END}")
    if right:
        components.append(right)
    return " ".join(" ".join(components).split())


def prepare_input(
    clinical_text: str,
    entity_text: str,
    section: str,
    occurrence: int,
    context_words: int,
) -> PreparedInput:
    text = str(clinical_text).strip()
    entity = str(entity_text).strip()
    if not text:
        raise ValueError("Enter a clinical text sample.")
    if not entity:
        raise ValueError("Enter the entity to classify.")
    occurrences = find_occurrences(text, entity)
    if not occurrences:
        raise ValueError(
            f"The entity {entity!r} was not found in the clinical text. "
            "Use the exact wording from the text."
        )
    if occurrence < 1 or occurrence > len(occurrences):
        raise ValueError(
            f"Occurrence must be between 1 and {len(occurrences)} for this text."
        )
    start, end = occurrences[occurrence - 1]
    return PreparedInput(
        clinical_text=text,
        entity_text=text[start:end],
        section=normalise_section(section),
        occurrence=occurrence,
        entity_start=start,
        entity_end=end,
        training_text=make_training_text(
            text, start, end, section, context_words
        ),
    )


def setfit_label_order(model: SetFitModel, metadata_labels: list[str]) -> list[str]:
    head = getattr(model, "model_head", None)
    classes = getattr(head, "classes_", None)
    if classes is None:
        return metadata_labels
    output: list[str] = []
    for value in np.asarray(classes).tolist():
        if isinstance(value, str):
            output.append(value.upper())
        else:
            output.append(metadata_labels[int(value)])
    return output


def setfit_probabilities(
    bundle: SetFitBundle, texts: list[str]
) -> np.ndarray:
    values = bundle.model.predict_proba(texts)
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)
    if probabilities.shape[1] != len(bundle.labels):
        raise RuntimeError(
            "SetFit probability columns do not match the saved label count."
        )
    return probabilities


def normalise_vectors(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def setfit_encode(model: SetFitModel, texts: list[str]) -> np.ndarray:
    values = model.model_body.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return normalise_vectors(np.asarray(values, dtype=np.float32))


def validate_development_csv(
    path: Path | None,
    label_column: str,
    allowed_labels: list[str],
) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Development CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {"training_text", label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["training_text", label_column]).copy()
    frame["training_text"] = frame["training_text"].astype(str).str.strip()
    frame[label_column] = frame[label_column].astype(str).str.upper().str.strip()
    frame = frame.loc[
        frame["training_text"].ne("")
        & frame[label_column].isin(allowed_labels)
    ].reset_index(drop=True)
    return frame


@st.cache_resource(show_spinner="Loading SetFit entity-type model…")
def load_setfit_bundle(
    model_dir_text: str,
    development_csv_text: str | None,
    device_text: str,
) -> SetFitBundle:
    model_dir = Path(model_dir_text)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"SetFit model directory not found: {model_dir}")
    metadata = read_json(model_dir / "training_metadata.json")
    if str(metadata.get("task", "")).lower() != "entity_type":
        raise ValueError("The configured SetFit model is not the entity_type model.")
    metadata_labels = [str(value).upper() for value in metadata.get("labels", [])]
    if sorted(metadata_labels) != sorted(ENTITY_TYPES):
        raise ValueError(
            f"Unexpected SetFit labels: {metadata_labels}; expected {ENTITY_TYPES}."
        )
    model = SetFitModel.from_pretrained(str(model_dir))
    device = choose_device(device_text)
    model.model_body.to(str(device))
    labels = setfit_label_order(model, metadata_labels)
    development_path = Path(development_csv_text) if development_csv_text else None
    development = validate_development_csv(
        development_path, "entity_type", ENTITY_TYPES
    )
    embeddings = None
    if development is not None and not development.empty:
        embeddings = setfit_encode(model, development["training_text"].tolist())
    return SetFitBundle(model, labels, development, embeddings)


def tokenize_for_protonet(
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.inference_mode()
def protonet_embeddings(bundle: ProtoNetBundle, texts: list[str]) -> torch.Tensor:
    if not texts:
        return torch.empty((0, bundle.prototypes.shape[1]))
    parts: list[torch.Tensor] = []
    batch_size = 32 if bundle.device.type == "cuda" else 12
    for offset in range(0, len(texts), batch_size):
        batch_texts = texts[offset : offset + batch_size]
        batch = tokenize_for_protonet(
            bundle.tokenizer,
            batch_texts,
            bundle.max_length,
            bundle.device,
        )
        embeddings = bundle.model(batch, bundle.start_id, bundle.end_id)
        parts.append(embeddings.detach().cpu())
    return torch.cat(parts, dim=0)


def protonet_scores(
    bundle: ProtoNetBundle, texts: list[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = protonet_embeddings(bundle, texts)
    prototypes = bundle.prototypes.detach().cpu()
    distances = torch.cdist(embeddings, prototypes, p=2).pow(2)
    logits = -distances / bundle.temperature
    probabilities = torch.softmax(logits, dim=-1)
    return embeddings, distances, logits, probabilities


@st.cache_resource(show_spinner="Loading BioClinicalBERT ProtoNet…")
def load_protonet_bundle(
    model_dir_text: str,
    development_csv_text: str | None,
    device_text: str,
    configured_max_length: int,
) -> ProtoNetBundle:
    model_dir = Path(model_dir_text)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"ProtoNet model directory not found: {model_dir}")
    metadata = read_json(model_dir / "protonet_context_metadata.json")
    labels = [str(value).upper() for value in metadata.get("labels", [])]
    if labels != CONTEXT_LABELS:
        raise ValueError(
            f"Unexpected ProtoNet label order: {labels}; expected {CONTEXT_LABELS}."
        )
    device = choose_device(device_text)
    encoder_dir = model_dir / "encoder"
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), use_fast=True)
    model = MentionEncoder(
        str(encoder_dir), projection_dim=int(metadata["projection_dim"])
    )
    projection_state = torch.load(
        model_dir / "projection.pt", map_location="cpu", weights_only=True
    )
    model.projection.load_state_dict(projection_state, strict=True)
    model.to(device)
    model.eval()
    payload = torch.load(
        model_dir / "class_prototypes.pt", map_location="cpu", weights_only=True
    )
    if [str(value).upper() for value in payload.get("labels", [])] != labels:
        raise ValueError("ProtoNet prototype labels do not match the metadata.")
    prototypes = payload.get("prototypes")
    expected_shape = (len(labels), int(metadata["projection_dim"]))
    if not isinstance(prototypes, torch.Tensor) or tuple(prototypes.shape) != expected_shape:
        raise ValueError(
            f"Expected prototype shape {expected_shape}; found "
            f"{getattr(prototypes, 'shape', None)}."
        )
    start_id = tokenizer.convert_tokens_to_ids(ENTITY_START)
    end_id = tokenizer.convert_tokens_to_ids(ENTITY_END)
    if start_id == tokenizer.unk_token_id or end_id == tokenizer.unk_token_id:
        raise RuntimeError("The frozen tokenizer does not contain the entity markers.")
    max_length = int(metadata.get("max_length", configured_max_length))
    development_path = Path(development_csv_text) if development_csv_text else None
    development = validate_development_csv(
        development_path, "context_label", CONTEXT_LABELS
    )
    bundle = ProtoNetBundle(
        tokenizer=tokenizer,
        model=model,
        prototypes=prototypes.float().to(device),
        labels=labels,
        temperature=float(metadata.get("temperature", 1.0)),
        start_id=int(start_id),
        end_id=int(end_id),
        max_length=max_length,
        device=device,
        development=development,
        development_embeddings=None,
    )
    if development is not None and not development.empty:
        bundle.development_embeddings = protonet_embeddings(
            bundle, development["training_text"].tolist()
        )
    return bundle


def load_models(config: AppConfig) -> tuple[SetFitBundle, ProtoNetBundle]:
    setfit = load_setfit_bundle(
        str(config.setfit_model_dir),
        str(config.setfit_development_csv)
        if config.setfit_development_csv
        else None,
        config.device,
    )
    protonet = load_protonet_bundle(
        str(config.protonet_model_dir),
        str(config.protonet_development_csv)
        if config.protonet_development_csv
        else None,
        config.device,
        config.max_length,
    )
    return setfit, protonet


def protected_word_variants(
    training_text: str,
    replacement: str | None,
    maximum: int,
) -> list[dict[str, Any]]:
    words = training_text.split()
    inside = False
    candidates: list[int] = []
    for index, word in enumerate(words):
        if word == ENTITY_START:
            inside = True
            continue
        if word == ENTITY_END:
            inside = False
            continue
        if not inside:
            candidates.append(index)
    if len(candidates) > maximum:
        step = len(candidates) / maximum
        candidates = [candidates[min(int(i * step), len(candidates) - 1)] for i in range(maximum)]
        candidates = list(dict.fromkeys(candidates))
    output: list[dict[str, Any]] = []
    for position in candidates:
        variant = words.copy()
        token = variant[position]
        if replacement is None:
            del variant[position]
        else:
            variant[position] = replacement
        output.append(
            {
                "position": position,
                "token": token,
                "text": " ".join(variant),
            }
        )
    return output


def explain_setfit(
    bundle: SetFitBundle,
    training_text: str,
    predicted_index: int,
    max_words: int,
) -> pd.DataFrame:
    original = setfit_probabilities(bundle, [training_text])[0]
    variants = protected_word_variants(training_text, None, max_words)
    if not variants:
        return pd.DataFrame()
    probabilities = setfit_probabilities(
        bundle, [item["text"] for item in variants]
    )
    rows = []
    for item, values in zip(variants, probabilities):
        rows.append(
            {
                "word": item["token"],
                "position": item["position"],
                "score_drop": float(
                    original[predicted_index] - values[predicted_index]
                ),
                "predicted_after_removal": bundle.labels[int(values.argmax())],
                "prediction_changed": int(values.argmax()) != predicted_index,
            }
        )
    return pd.DataFrame(rows).sort_values("score_drop", ascending=False)


def explain_protonet_occlusion(
    bundle: ProtoNetBundle,
    training_text: str,
    predicted_index: int,
    runner_index: int,
    original_logits: torch.Tensor,
    max_words: int,
) -> pd.DataFrame:
    replacement = bundle.tokenizer.mask_token or "[MASK]"
    variants = protected_word_variants(training_text, replacement, max_words)
    if not variants:
        return pd.DataFrame()
    _, _, logits, probabilities = protonet_scores(
        bundle, [item["text"] for item in variants]
    )
    original_margin = float(
        original_logits[predicted_index] - original_logits[runner_index]
    )
    rows = []
    for item, values, probability_values in zip(variants, logits, probabilities):
        new_margin = float(values[predicted_index] - values[runner_index])
        new_index = int(probability_values.argmax())
        rows.append(
            {
                "word": item["token"],
                "position": item["position"],
                "margin_drop": original_margin - new_margin,
                "predicted_after_masking": bundle.labels[new_index],
                "prediction_changed": new_index != predicted_index,
            }
        )
    return pd.DataFrame(rows).sort_values("margin_drop", ascending=False)


def nearest_setfit_examples(
    bundle: SetFitBundle,
    training_text: str,
    predicted_label: str,
    runner_label: str,
    top_k: int,
) -> tuple[pd.DataFrame, float | None]:
    if bundle.development is None or bundle.development_embeddings is None:
        return pd.DataFrame(), None
    query = setfit_encode(bundle.model, [training_text])[0]
    similarities = bundle.development_embeddings @ query
    rows: list[pd.DataFrame] = []
    nearest_distance: float | None = None
    for role, label in (("predicted", predicted_label), ("runner_up", runner_label)):
        mask = bundle.development["entity_type"].eq(label).to_numpy()
        candidate_indices = np.flatnonzero(mask)
        if not len(candidate_indices):
            continue
        selected = candidate_indices[
            np.argsort(similarities[candidate_indices])[::-1][:top_k]
        ]
        if role == "predicted":
            nearest_distance = float(1.0 - similarities[selected[0]])
        part = bundle.development.iloc[selected].copy()
        part.insert(0, "role", role)
        part.insert(1, "similarity", similarities[selected])
        rows.append(part)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), nearest_distance


def nearest_protonet_examples(
    bundle: ProtoNetBundle,
    query_embedding: torch.Tensor,
    predicted_label: str,
    runner_label: str,
    top_k: int,
) -> tuple[pd.DataFrame, float | None]:
    if bundle.development is None or bundle.development_embeddings is None:
        return pd.DataFrame(), None
    distances = torch.cdist(
        query_embedding.reshape(1, -1), bundle.development_embeddings, p=2
    ).pow(2)[0]
    rows: list[pd.DataFrame] = []
    nearest_distance: float | None = None
    for role, label in (("predicted", predicted_label), ("runner_up", runner_label)):
        mask = bundle.development["context_label"].eq(label).to_numpy()
        candidate_indices = np.flatnonzero(mask)
        if not len(candidate_indices):
            continue
        candidate_distances = distances[candidate_indices]
        order = torch.argsort(candidate_distances)[:top_k].numpy()
        selected = candidate_indices[order]
        if role == "predicted":
            nearest_distance = float(distances[selected[0]])
        part = bundle.development.iloc[selected].copy()
        part.insert(0, "role", role)
        part.insert(1, "squared_distance", distances[selected].numpy())
        rows.append(part)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), nearest_distance


def classify_one(
    prepared: PreparedInput,
    setfit: SetFitBundle,
    protonet: ProtoNetBundle,
    config: AppConfig,
) -> dict[str, Any]:
    setfit_probs = setfit_probabilities(setfit, [prepared.training_text])[0]
    setfit_order = np.argsort(setfit_probs)[::-1]
    entity_index = int(setfit_order[0])
    entity_runner_index = int(setfit_order[1])
    entity_label = setfit.labels[entity_index]
    entity_runner = setfit.labels[entity_runner_index]

    embeddings, distances, logits, probabilities = protonet_scores(
        protonet, [prepared.training_text]
    )
    context_order = torch.argsort(probabilities[0], descending=True)
    context_index = int(context_order[0])
    context_runner_index = int(context_order[1])
    context_label = protonet.labels[context_index]
    context_runner = protonet.labels[context_runner_index]
    prototype_margin = float(
        distances[0, context_runner_index] - distances[0, context_index]
    )
    logit_margin = float(
        logits[0, context_index] - logits[0, context_runner_index]
    )

    setfit_explanation = explain_setfit(
        setfit,
        prepared.training_text,
        entity_index,
        config.max_explanation_words,
    )
    protonet_explanation = explain_protonet_occlusion(
        protonet,
        prepared.training_text,
        context_index,
        context_runner_index,
        logits[0],
        config.max_explanation_words,
    )
    setfit_neighbours, setfit_nearest_distance = nearest_setfit_examples(
        setfit,
        prepared.training_text,
        entity_label,
        entity_runner,
        config.top_k_examples,
    )
    protonet_neighbours, protonet_nearest_distance = nearest_protonet_examples(
        protonet,
        embeddings[0],
        context_label,
        context_runner,
        config.top_k_examples,
    )

    reasons: list[str] = []
    high = False
    moderate = False
    entity_gap = float(setfit_probs[entity_index] - setfit_probs[entity_runner_index])
    if entity_gap < 0.10:
        moderate = True
        reasons.append("The two highest SetFit scores are close.")
    if logit_margin < 0.50:
        moderate = True
        reasons.append("The ProtoNet decision margin is small.")
    if context_label == "UNCERTAIN":
        moderate = True
        reasons.append("UNCERTAIN predictions are prioritised for review.")
    if not setfit_explanation.empty and setfit_explanation["prediction_changed"].any():
        high = True
        reasons.append("Removing one context word changes the SetFit prediction.")
    if not protonet_explanation.empty and protonet_explanation["prediction_changed"].any():
        high = True
        reasons.append("Masking one context word changes the ProtoNet prediction.")
    review_priority = "HIGH" if high else "MODERATE" if moderate else "LOW"
    if not reasons:
        reasons.append("Both models are stable under the current exploratory checks.")

    return {
        "prepared": prepared,
        "entity_label": entity_label,
        "entity_runner": entity_runner,
        "entity_probabilities": setfit_probs,
        "entity_gap": entity_gap,
        "context_label": context_label,
        "context_runner": context_runner,
        "context_probabilities": probabilities[0].numpy(),
        "prototype_distances": distances[0].numpy(),
        "prototype_margin": prototype_margin,
        "logit_margin": logit_margin,
        "setfit_explanation": setfit_explanation,
        "protonet_explanation": protonet_explanation,
        "setfit_neighbours": setfit_neighbours,
        "protonet_neighbours": protonet_neighbours,
        "setfit_nearest_distance": setfit_nearest_distance,
        "protonet_nearest_distance": protonet_nearest_distance,
        "review_priority": review_priority,
        "review_reasons": reasons,
    }


def probability_frame(labels: list[str], values: Iterable[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": labels,
            "score": [float(value) for value in values],
            "colour": [CLASS_COLOURS.get(label, "#8b7cff") for label in labels],
        }
    ).sort_values("score", ascending=False)


def score_chart(frame: pd.DataFrame, title: str, x_title: str = "Model score") -> alt.Chart:
    return (
        alt.Chart(frame, title=title)
        .mark_bar(cornerRadiusEnd=10, height=34)
        .encode(
            x=alt.X("score:Q", title=x_title, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(
                "class:N",
                sort="-x",
                title=None,
                scale=alt.Scale(paddingInner=0.38, paddingOuter=0.28),
            ),
            color=alt.Color("colour:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("class:N", title="Class"),
                alt.Tooltip("score:Q", title="Score", format=".3f"),
            ],
        )
        .properties(height=max(260, 70 * len(frame)))
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#cbd3e5",
            titleColor="#a8b0c3",
            gridColor="#263047",
            labelFontSize=13,
            titleFontSize=13,
            labelPadding=8,
            titlePadding=16,
        )
        .configure_title(color="#f4f7ff", anchor="start", fontSize=16, offset=14)
    )


def distance_chart(labels: list[str], distances: np.ndarray) -> alt.Chart:
    frame = pd.DataFrame(
        {
            "class": labels,
            "distance": distances.astype(float),
            "colour": [CLASS_COLOURS[label] for label in labels],
        }
    ).sort_values("distance")
    return (
        alt.Chart(frame, title="Distance from each learned prototype")
        .mark_bar(cornerRadiusEnd=10, height=36)
        .encode(
            x=alt.X("distance:Q", title="Squared Euclidean distance"),
            y=alt.Y(
                "class:N",
                sort="x",
                title=None,
                scale=alt.Scale(paddingInner=0.38, paddingOuter=0.30),
            ),
            color=alt.Color("colour:N", scale=None, legend=None),
            tooltip=["class:N", alt.Tooltip("distance:Q", format=".4f")],
        )
        .properties(height=max(300, 72 * len(frame)))
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#cbd3e5",
            titleColor="#a8b0c3",
            gridColor="#263047",
            labelFontSize=13,
            titleFontSize=13,
            labelPadding=8,
            titlePadding=16,
        )
        .configure_title(color="#f4f7ff", anchor="start", fontSize=16, offset=14)
    )


def sensitivity_chart(
    frame: pd.DataFrame, value_column: str, title: str
) -> alt.Chart | None:
    if frame.empty:
        return None
    chart_frame = frame.nlargest(10, value_column).copy()
    chart_frame["direction"] = np.where(
        chart_frame[value_column] >= 0, "supports prediction", "opposes prediction"
    )
    return (
        alt.Chart(chart_frame, title=title)
        .mark_bar(cornerRadiusEnd=8, height=28)
        .encode(
            x=alt.X(f"{value_column}:Q", title="Change after removal or masking"),
            y=alt.Y(
                "word:N",
                sort="-x",
                title=None,
                scale=alt.Scale(paddingInner=0.34, paddingOuter=0.24),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["supports prediction", "opposes prediction"],
                    range=["#31d7c6", "#ff6685"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["word:N", alt.Tooltip(f"{value_column}:Q", format=".4f")],
        )
        .properties(height=max(420, 56 * len(chart_frame)))
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#cbd3e5",
            titleColor="#a8b0c3",
            gridColor="#263047",
            labelFontSize=13,
            titleFontSize=13,
            labelPadding=8,
            titlePadding=16,
        )
        .configure_title(color="#f4f7ff", anchor="start", fontSize=16, offset=14)
        .configure_legend(labelColor="#cbd3e5", labelFontSize=13, symbolSize=160)
    )


def render_card(label: str, value: str, detail: str, accent: str = "violet") -> None:
    st.markdown(
        f"""
        <div class="glass-card accent-{html.escape(accent)}">
          <div class="label">{html.escape(label)}</div>
          <div class="value">{html.escape(value)}</div>
          <div class="detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def highlighted_source(prepared: PreparedInput) -> str:
    before = html.escape(prepared.clinical_text[: prepared.entity_start])
    entity = html.escape(prepared.clinical_text[prepared.entity_start : prepared.entity_end])
    after = html.escape(prepared.clinical_text[prepared.entity_end :])
    return f'{before}<span class="entity-chip">{entity}</span>{after}'


def render_neighbours(
    frame: pd.DataFrame,
    label_column: str,
    distance_column: str,
) -> None:
    if frame.empty:
        st.info("Add the relevant development CSV in the configuration to show similar examples.")
        return
    for _, row in frame.iterrows():
        role = "Supports prediction" if row["role"] == "predicted" else "Closest counterexample"
        metric = float(row[distance_column])
        with st.container(border=True):
            st.caption(
                f"{role} · {row[label_column]} · {distance_column.replace('_', ' ')} {metric:.4f}"
            )
            st.write(str(row["training_text"]))


def result_export(result: dict[str, Any], setfit: SetFitBundle, protonet: ProtoNetBundle) -> dict[str, Any]:
    prepared: PreparedInput = result["prepared"]
    return {
        "entity_text": prepared.entity_text,
        "section": prepared.section,
        "entity_start": prepared.entity_start,
        "entity_end": prepared.entity_end,
        "model_input": prepared.training_text,
        "setfit_entity_type": result["entity_label"],
        "setfit_model_score": float(max(result["entity_probabilities"])),
        "setfit_runner_up": result["entity_runner"],
        "setfit_scores": {
            label: float(value)
            for label, value in zip(setfit.labels, result["entity_probabilities"])
        },
        "protonet_context": result["context_label"],
        "protonet_model_score": float(max(result["context_probabilities"])),
        "protonet_runner_up": result["context_runner"],
        "prototype_distance_margin": float(result["prototype_margin"]),
        "protonet_scores": {
            label: float(value)
            for label, value in zip(protonet.labels, result["context_probabilities"])
        },
        "prototype_distances": {
            label: float(value)
            for label, value in zip(protonet.labels, result["prototype_distances"])
        },
        "review_priority": result["review_priority"],
        "review_reasons": result["review_reasons"],
        "disclaimer": "Model scores are not calibrated probabilities of correctness.",
    }


def render_live_result(
    result: dict[str, Any],
    setfit: SetFitBundle,
    protonet: ProtoNetBundle,
) -> None:
    prepared: PreparedInput = result["prepared"]
    st.markdown("### Combined classification")
    columns = st.columns(3)
    with columns[0]:
        render_card(
            "SetFit · entity type",
            result["entity_label"],
            f"Runner-up: {result['entity_runner']}",
            "violet",
        )
    with columns[1]:
        render_card(
            "BioClinicalBERT ProtoNet · context",
            result["context_label"],
            f"Runner-up: {result['context_runner']}",
            "teal",
        )
    with columns[2]:
        priority = result["review_priority"]
        priority_class = priority.lower()
        st.markdown(
            f"""
            <div class="glass-card">
              <div class="label">Exploratory review signal</div>
              <div class="value review-{priority_class}">{priority}</div>
              <div class="detail">Not an automated clinical decision</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="input-preview">{highlighted_source(prepared)}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Inspect the exact model input"):
        st.code(prepared.training_text, language="text")

    score_left, score_right = st.columns(2)
    with score_left:
        st.altair_chart(
            score_chart(
                probability_frame(setfit.labels, result["entity_probabilities"]),
                "SetFit entity-type scores",
            ),
            use_container_width=True,
        )
    with score_right:
        st.altair_chart(
            score_chart(
                probability_frame(protonet.labels, result["context_probabilities"]),
                "ProtoNet context scores",
            ),
            use_container_width=True,
        )

    st.markdown(
        '<div class="explanation-note">Model scores show relative model preference. '
        "They are not calibrated probabilities that a prediction is clinically correct.</div>",
        unsafe_allow_html=True,
    )

    explanation_tabs = st.tabs(
        ["SetFit explanation", "ProtoNet explanation", "Similar examples", "Review signal"]
    )
    with explanation_tabs[0]:
        st.markdown("#### Why SetFit selected the entity type")
        st.write(
            f"SetFit assigned **{result['entity_label']}**. The closest competing class was "
            f"**{result['entity_runner']}**, with a top-two score gap of "
            f"**{result['entity_gap']:.3f}**."
        )
        chart = sensitivity_chart(
            result["setfit_explanation"],
            "score_drop",
            "Words whose removal most reduced the selected SetFit score",
        )
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        with st.expander("Full SetFit word-removal table"):
            st.dataframe(result["setfit_explanation"], use_container_width=True)

    with explanation_tabs[1]:
        st.markdown("#### Why ProtoNet selected the clinical context")
        st.write(
            f"The entity embedding was closest to the **{result['context_label']}** "
            f"prototype. Its runner-up was **{result['context_runner']}**. The "
            f"predicted-versus-runner-up prototype-distance margin was "
            f"**{result['prototype_margin']:.4f}**."
        )
        st.altair_chart(
            distance_chart(protonet.labels, result["prototype_distances"]),
            use_container_width=True,
        )
        chart = sensitivity_chart(
            result["protonet_explanation"],
            "margin_drop",
            "Words whose masking most reduced the ProtoNet decision margin",
        )
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        with st.expander("Full ProtoNet word-masking table"):
            st.dataframe(result["protonet_explanation"], use_container_width=True)

    with explanation_tabs[2]:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Similar SetFit examples")
            render_neighbours(
                result["setfit_neighbours"], "entity_type", "similarity"
            )
        with right:
            st.markdown("#### Similar ProtoNet examples")
            render_neighbours(
                result["protonet_neighbours"],
                "context_label",
                "squared_distance",
            )

    with explanation_tabs[3]:
        st.markdown(f"#### Review priority: {result['review_priority']}")
        for reason in result["review_reasons"]:
            st.write(f"- {reason}")
        st.caption(
            "The thresholds are exploratory interface rules, not clinically validated "
            "accept/reject thresholds. Review priority must not be used as an automated decision."
        )

    payload = result_export(result, setfit, protonet)
    st.download_button(
        "Download this result as JSON",
        data=json.dumps(payload, indent=2),
        file_name="clinical_entity_classification.json",
        mime="application/json",
        use_container_width=False,
    )


def batch_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "example_001",
                "clinical_text": "The patient previously received treatment for depression and is currently stable.",
                "entity_text": "depression",
                "section": "past_medical_history",
                "occurrence": 1,
            }
        ]
    )


def classify_batch(
    uploaded: pd.DataFrame,
    setfit: SetFitBundle,
    protonet: ProtoNetBundle,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"clinical_text", "entity_text", "section"}
    missing = required.difference(uploaded.columns)
    if missing:
        raise ValueError(f"Uploaded CSV is missing columns: {sorted(missing)}")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row_number, row in uploaded.iterrows():
        sample_id = str(row.get("sample_id", f"row_{row_number + 1}"))
        try:
            occurrence_value = row.get("occurrence", 1)
            occurrence = 1 if pd.isna(occurrence_value) else int(occurrence_value)
            prepared = prepare_input(
                str(row["clinical_text"]),
                str(row["entity_text"]),
                str(row["section"]),
                occurrence,
                config.context_words,
            )
            setfit_probs = setfit_probabilities(setfit, [prepared.training_text])[0]
            _, distances, logits, proto_probs = protonet_scores(
                protonet, [prepared.training_text]
            )
            setfit_order = np.argsort(setfit_probs)[::-1]
            proto_order = torch.argsort(proto_probs[0], descending=True)
            context_index = int(proto_order[0])
            context_runner = int(proto_order[1])
            results.append(
                {
                    "sample_id": sample_id,
                    "entity_text": prepared.entity_text,
                    "section": prepared.section,
                    "setfit_entity_type": setfit.labels[int(setfit_order[0])],
                    "setfit_model_score": float(setfit_probs[int(setfit_order[0])]),
                    "setfit_runner_up": setfit.labels[int(setfit_order[1])],
                    "protonet_context": protonet.labels[context_index],
                    "protonet_model_score": float(proto_probs[0, context_index]),
                    "protonet_runner_up": protonet.labels[context_runner],
                    "prototype_distance_margin": float(
                        distances[0, context_runner] - distances[0, context_index]
                    ),
                    "decision_logit_margin": float(
                        logits[0, context_index] - logits[0, context_runner]
                    ),
                    "model_input": prepared.training_text,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "row_number": int(row_number) + 2,
                    "sample_id": sample_id,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(results), pd.DataFrame(errors)


def render_batch_page(
    setfit: SetFitBundle,
    protonet: ProtoNetBundle,
    config: AppConfig,
) -> None:
    st.markdown("## Batch classification")
    st.write(
        "Upload a local CSV to classify several already identified entity mentions. "
        "The file is processed in the current application session."
    )
    template = batch_template()
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False),
        "clinical_classifier_batch_template.csv",
        "text/csv",
    )
    upload = st.file_uploader("Upload batch CSV", type=["csv"])
    if upload is None:
        return
    frame = pd.read_csv(upload)
    st.dataframe(frame.head(20), use_container_width=True)
    if st.button("Classify batch", type="primary", use_container_width=True):
        with st.spinner("Running both classifiers…"):
            results, errors = classify_batch(frame, setfit, protonet, config)
        st.session_state["batch_results"] = results
        st.session_state["batch_errors"] = errors
    results = st.session_state.get("batch_results")
    errors = st.session_state.get("batch_errors")
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.success(f"Classified {len(results)} entity mentions.")
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download batch results",
            results.to_csv(index=False),
            "clinical_classifier_batch_results_local_only.csv",
            "text/csv",
        )
    if isinstance(errors, pd.DataFrame) and not errors.empty:
        st.warning(f"{len(errors)} rows could not be classified.")
        st.dataframe(errors, use_container_width=True)


def render_model_page(
    config: AppConfig,
    setfit: SetFitBundle,
    protonet: ProtoNetBundle,
) -> None:
    st.markdown("## Model and responsible-use information")
    left, right = st.columns(2)
    with left:
        render_card(
            "Task 1",
            "SetFit",
            "Entity type: diagnosis, medication or symptom",
            "violet",
        )
    with right:
        render_card(
            "Task 2",
            "BioClinicalBERT ProtoNet",
            "Context: historical, negated, present or uncertain",
            "teal",
        )
    st.markdown("### Processing sequence")
    st.markdown(
        """
        1. The user supplies text and identifies one entity mention.
        2. The application builds the same 15-word marked input used in model development.
        3. SetFit predicts the entity type.
        4. BioClinicalBERT ProtoNet independently predicts the entity context.
        5. The interface reports each decision separately and adds local explanations.
        """
    )
    st.markdown("### Important limitations")
    st.markdown(
        """
        - This is a research prototype, not a medical device or autonomous clinical decision system.
        - A model score is not a calibrated probability that a prediction is correct.
        - Explanations describe model behaviour; they do not prove clinical validity.
        - Similar examples may contain sensitive text. Keep the application and outputs in an approved local environment.
        - The review signal is exploratory and must not automatically accept, reject or prioritise patient care.
        """
    )
    with st.expander("Loaded configuration"):
        st.json(
            {
                "setfit_model_dir": str(config.setfit_model_dir),
                "protonet_model_dir": str(config.protonet_model_dir),
                "setfit_development_csv": str(config.setfit_development_csv),
                "protonet_development_csv": str(config.protonet_development_csv),
                "device": str(protonet.device),
                "context_words_each_side": config.context_words,
                "protonet_max_length": protonet.max_length,
                "setfit_labels": setfit.labels,
                "protonet_labels": protonet.labels,
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="Clinical Context Classifier",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">FEW-SHOT CLINICAL NLP</div>
          <h1>Clinical Entity Intelligence</h1>
          <p>Two independent classification tasks in one transparent workspace: SetFit for entity type and BioClinicalBERT ProtoNet for contextual status.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        config = load_config()
        setfit, protonet = load_models(config)
    except Exception as exc:
        st.error("The application could not load its configured models.")
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.markdown("### Clinical Entity Intelligence")
        page = st.radio(
            "Workspace",
            ["Live analyser", "Batch classification", "Model information"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Compute device: {protonet.device.type.upper()}")
        st.caption(f"Context window: {config.context_words} words each side")
        st.caption("All inference runs locally in this Streamlit process.")

    if page == "Batch classification":
        render_batch_page(setfit, protonet, config)
        return
    if page == "Model information":
        render_model_page(config, setfit, protonet)
        return

    st.markdown("## Analyse one clinical entity")
    samples = {
        "Historical diagnosis": (
            "The patient previously received treatment for depression and is currently stable.",
            "depression",
            "past_medical_history",
        ),
        "Present medication": (
            "The patient is currently taking sertraline 50 mg once daily.",
            "sertraline",
            "medications",
        ),
        "Negated symptom": (
            "The patient denies chest pain, palpitations or shortness of breath.",
            "chest pain",
            "history_present_illness",
        ),
        "Uncertain diagnosis": (
            "The imaging appearances may represent early pneumonia.",
            "pneumonia",
            "investigations",
        ),
        "Custom": ("", "", "history_present_illness"),
    }
    selected_sample = st.selectbox("Start with a sample", list(samples))
    sample_text, sample_entity, sample_section = samples[selected_sample]
    clinical_text = st.text_area(
        "Clinical text",
        value=sample_text,
        height=150,
        placeholder="Paste a short clinical text sample…",
        key=f"clinical_text_{selected_sample}",
    )
    section = st.text_input(
        "Clinical section", value=sample_section,
        key=f"clinical_section_{selected_sample}",
    )
    st.caption("After editing the text, click outside the text box to update the selection area.")
    text = clinical_text.strip()
    revision = hashlib.sha256(
        (selected_sample + "\0" + text + "\0" + section).encode("utf-8")
    ).hexdigest()
    if st.session_state.get("highlight_revision") != revision:
        st.session_state["highlight_revision"] = revision
        st.session_state.pop("highlight_result", None)
        st.session_state.pop("highlight_signature", None)
    if not text:
        st.info("Paste clinical text above to begin.")
        return
    selection = entity_selector(text=text, revision=revision, key="entity_highlighter", default=None)
    try:
        prepared = prepare_selected_input(
            text, selection, revision, section, config.context_words
        )
        if prepared is None:
            st.session_state.pop("highlight_result", None)
            st.session_state.pop("highlight_signature", None)
            return
        st.markdown("**Selected entity**")
        st.write(prepared.entity_text)
        signature = (revision, prepared.entity_start, prepared.entity_end)
        if st.session_state.get("highlight_signature") != signature:
            st.session_state.pop("highlight_result", None)
            with st.spinner("Running SetFit and BioClinicalBERT ProtoNet…"):
                result = classify_one(prepared, setfit, protonet, config)
            st.session_state["highlight_result"] = result
            st.session_state["highlight_signature"] = signature
        result = st.session_state.get("highlight_result")
        if result is not None:
            st.divider()
            render_live_result(result, setfit, protonet)
    except Exception as exc:
        st.session_state.pop("highlight_result", None)
        st.session_state.pop("highlight_signature", None)
        st.error(str(exc))


if __name__ == "__main__":
    main()

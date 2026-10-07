"""The only supported SensiFake manual annotation application."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from manual_annotator.script.annotation_database import RUBRIC_FIELDS, SharedStore, identity
from manual_annotator.script.annotation_schema import (
    AnnotationError,
    calculate_score,
    sensitivity_level,
)
from manual_annotator.script.import_batches import SNAPSHOT, prepare_legacy, prepare_zip
from manual_annotator.script.paths import REPOSITORY_ROOT

MODEL_REVIEW_KEYS = st.components.v2.component(
    "model_review_keyboard_shortcuts",
    html="<span aria-hidden='true'></span>",
    js="""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const documentRef = parentElement.ownerDocument;
  const handleKey = (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.repeat) return;
    const target = event.target;
    if (target?.isContentEditable || target?.matches?.('textarea, select, input:not([type=radio])')) return;
    const allowed = ['1', '2', '3'];
    if (data.stage === 'reveal') allowed.push('Enter');
    if (!allowed.includes(event.key)) return;
    event.preventDefault();
    setTriggerValue('pressed', { key: event.key, case_id: data.case_id });
  };
  documentRef.addEventListener('keydown', handleKey);
  return () => documentRef.removeEventListener('keydown', handleKey);
}
""",
)


def model_review_queue(store: SharedStore, name: str, batch_id: str | None) -> None:
    person_key, _ = identity(name)
    state_key = f"lease_{store.path}_{person_key}_model_review"
    if state_key not in st.session_state:
        if st.button("Resume or reserve an image", type="primary"):
            lease = store.reserve(name, batch_id, kind="model_review")
            if lease:
                st.session_state[state_key] = lease
                st.rerun()
            st.info(
                "No pending model-review images are available. Check the batch and your assigned annotator name."
            )
        return

    lease = st.session_state[state_key]
    content_hash, token = lease["content_hash"], lease["token"]
    case = store.model_review_case(content_hash)
    provenance = case["provenance"]
    st.caption(f"Image SF-{content_hash[:20]}")
    if st.button("Release image"):
        store.release(token, name)
        del st.session_state[state_key]
        st.rerun()
    st.image(store.image(content_hash), width="stretch")

    draft = store.draft(lease) or {}
    initial = draft.get("human_initial_label")
    stage = "reveal" if initial else "human"
    shortcut = MODEL_REVIEW_KEYS(
        key=f"model_review_keys_{token}",
        data={"stage": stage, "case_id": token},
        on_pressed_change=lambda: None,
    )
    pressed = getattr(shortcut, "pressed", None)
    key = (
        pressed.get("key")
        if isinstance(pressed, Mapping) and pressed.get("case_id") == token
        else None
    )
    st.caption("1 Low · 2 Medium · 3 High · Enter Confirm and next (after reveal)")
    choices = {"1": "low", "2": "medium", "3": "high"}

    if not initial:
        st.write("**Step 1 — Your independent sensitivity decision**")
        st.caption("The model prediction is hidden until you save this first choice.")
        buttons = st.columns(3)
        selected = None
        for number, level in choices.items():
            if buttons[int(number) - 1].button(
                f"{number} {level.title()}", key=f"{token}_initial_{level}"
            ):
                selected = level
        selected = choices.get(key, selected)
        if selected:
            store.save_model_review_draft(token, name, initial=selected)
            st.rerun()
        return

    st.write("**Step 2 — Model prediction revealed**")
    prediction = provenance["model_prediction"]
    st.write(f"Your initial label: **{initial}** · Model prediction: **{prediction}**")
    st.write("**AGREEMENT**" if initial == prediction else "**DISAGREEMENT**")
    confidence = provenance.get("confidence", "")
    if confidence not in (None, ""):
        st.write(f"Model confidence: **{float(confidence):.3f}**")
    probabilities = {
        label: provenance.get(f"prob_{label}", "") for label in ("low", "medium", "high")
    }
    if any(value not in (None, "") for value in probabilities.values()):
        st.write("Model probabilities:", probabilities)
    st.caption(
        f"Dataset: {provenance.get('source_dataset', 'unknown')} · Source label: {provenance.get('normalized_label', 'unknown')}"
    )

    st.write("**Step 3 — Confirm or change your final label**")
    final_key = f"{token}_final_label"
    if final_key not in st.session_state:
        st.session_state[final_key] = draft.get("human_final_label", initial)
    if key in choices:
        st.session_state[final_key] = choices[key]
    final = st.radio("Final sensitivity", ("low", "medium", "high"), key=final_key, horizontal=True)
    store.save_model_review_draft(token, name, final=final)
    if st.button("Confirm and next", type="primary") or key == "Enter":
        store.complete_model_review(token, name, final)
        next_lease = store.reserve(name, batch_id, kind="model_review")
        if next_lease:
            st.session_state[state_key] = next_lease
        else:
            del st.session_state[state_key]
        st.rerun()


def rubric_inputs(defaults: dict, key: str) -> dict:
    columns = st.columns(3)
    public = columns[0].radio(
        "Public relevance · 0–2",
        (0, 1, 2),
        index=defaults.get("public_relevance", 0),
        key=f"{key}_public",
    )
    harm = columns[1].radio(
        "Harm urgency · 0–2", (0, 1, 2), index=defaults.get("harm_urgency", 0), key=f"{key}_harm"
    )
    vulnerability = columns[2].radio(
        "Vulnerability · 0–1",
        (0, 1),
        index=defaults.get("vulnerability", 0),
        key=f"{key}_vulnerability",
    )
    score = calculate_score(public, harm, vulnerability)
    st.write(f"Sensitivity: **{score}/5 · {sensitivity_level(score)}**")
    rationale = st.text_area(
        "Optional observable rationale",
        max_chars=280,
        value=defaults.get("sensitivity_rationale", ""),
        key=f"{key}_rationale",
    )
    confidence_key, review_key = f"{key}_confidence", f"{key}_needs_review"
    if review_key not in st.session_state:
        st.session_state[review_key] = defaults.get("needs_review", False)

    def flag_low_confidence():
        if st.session_state[confidence_key] == "low":
            st.session_state[review_key] = True

    confidence = st.selectbox(
        "Annotation confidence",
        ("high", "medium", "low"),
        index=("high", "medium", "low").index(defaults.get("annotation_confidence", "medium")),
        key=confidence_key,
        on_change=flag_low_confidence,
    )
    needs_review = st.checkbox(
        "Needs review",
        key=review_key,
        help="Selected when confidence changes to low; you may edit the flag.",
    )
    return {
        "public_relevance": public,
        "harm_urgency": harm,
        "vulnerability": vulnerability,
        "sensitivity_rationale": rationale,
        "annotation_confidence": confidence,
        "needs_review": needs_review,
    }


def work_queue(store: SharedStore, name: str, batch_id: str | None, kind: str):
    person_key, _ = identity(name)
    state_key = f"lease_{store.path}_{person_key}_{kind}"
    include_reviewed = (
        kind == "review"
        and not store.is_review_snapshot
        and st.checkbox("Include previously reviewed images")
    )
    if state_key not in st.session_state:
        if st.button("Resume or reserve an image", type="primary"):
            lease = store.reserve(name, batch_id, kind=kind, include_reviewed=include_reviewed)
            if lease:
                st.session_state[state_key] = lease
                st.rerun()
            else:
                st.info("No eligible images are available. Others may have active reservations.")
        return
    lease = st.session_state[state_key]
    content_hash, token = lease["content_hash"], lease["token"]
    st.caption(f"Image SF-{content_hash[:20]}")
    st.caption(
        "The current reservation stays selected until you save or release it, even if you change the batch filter."
    )
    if st.button("Release image"):
        store.release(token, name)
        del st.session_state[state_key]
        st.rerun()
    st.image(store.image(content_hash), width="stretch")
    decision = store.decision(content_hash) if kind == "review" else None
    defaults = store.draft(lease) or (decision["current"] if decision else {})
    action, reason = "confirm", ""
    if decision:
        st.write(f"Original annotator: **{decision['annotator']}**")
        st.json({key: decision["original"][key] for key in RUBRIC_FIELDS})
        if decision["reviews"]:
            st.write("Latest reviewed annotation")
            st.json({key: decision["current"][key] for key in RUBRIC_FIELDS})
        action = st.radio(
            "Review decision",
            ("confirm", "correct"),
            format_func=lambda value: (
                "Confirm current annotation" if value == "confirm" else "Correct annotation"
            ),
            key=f"{token}_action",
        )
    if not decision or action == "correct":
        values = rubric_inputs(defaults, token)
    else:
        values = decision["current"]
    if action == "correct":
        reason = st.text_area(
            "Reason for correction (required)",
            value=defaults.get("review_reason", ""),
            key=f"{token}_reason",
        )
    # Widget changes trigger reruns; persist drafts and renew only a still-valid lease.
    try:
        store.save_draft(token, name, {**values, "review_reason": reason})
    except AnnotationError as exc:
        st.warning(str(exc))
        if st.button("Return to queue"):
            del st.session_state[state_key]
            st.rerun()
        return
    st.caption(
        f"Draft saved when controls change. Reservations expire after {store.reservation_seconds // 60} minutes without interaction. Text is saved after leaving the field."
    )
    if st.button(
        "Save completed annotation" if kind == "annotation" else "Save review", type="primary"
    ):
        store.complete(token, name, values, action=action, reason=reason)
        del st.session_state[state_key]
        st.success("Saved durably. Reserve the next image when ready.")
        st.rerun()


def imports(store: SharedStore):
    st.write(
        "Import one ZIP containing images and, optionally, a root metadata.json mapping filenames to provenance. A model-review metadata.json is detected automatically. All images must be readable."
    )
    name = st.text_input("Batch name")
    role = st.selectbox("Dataset role", ("custom", "rrdataset", "gold_development", "human_train"))
    upload = st.file_uploader("Image ZIP", type="zip")
    if st.button("Import batch", disabled=upload is None):
        with st.spinner("Validating images and saving batch…"):
            result = store.import_batches(prepare_zip(upload, name, role))
        st.success(
            f"Imported {result['entries']} filenames: {result['new_images']} new images, {result['duplicate_images']} duplicate images."
        )
        if result["duplicates"]:
            st.dataframe(result["duplicates"], hide_index=True)
    with st.expander("Import existing completed annotations"):
        st.write(
            "Imports only the validated 401-row snapshot into separate gold_development and human_train batches. Existing source files are preserved. Imported annotations await independent review."
        )
        st.caption(SNAPSHOT)
        if st.button("Import validated 401-row snapshot"):
            with st.spinner("Checking the snapshot and copying verified images…"):
                store.import_batches(prepare_legacy(REPOSITORY_ROOT))
            st.success("Imported 401 completed annotations.")
    st.write("Batch availability")
    for batch in store.batches():
        opened = st.checkbox(
            batch["name"], value=bool(batch["is_open"]), key=f"open_{batch['batch_id']}"
        )
        if opened != bool(batch["is_open"]):
            store.set_open(batch["batch_id"], opened)
    st.caption(
        "Unchecked batches leave the queues. Existing reservations may finish; exports remain available."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage",
        type=Path,
        default=os.environ.get("SENSIFAKE_STORAGE")
        or REPOSITORY_ROOT / "manual_annotator" / "data" / "sensifake.sqlite3",
    )
    parser.add_argument("--reservation-minutes", type=int, default=15)
    args = parser.parse_args()
    st.set_page_config(page_title="SensiFake shared annotator", layout="wide")
    st.title("SensiFake · Manual sensitivity annotation")
    if not args.storage.is_absolute() or args.reservation_minutes < 1:
        st.error(
            "Storage must be an absolute SQLite file path. Reservation minutes must be positive."
        )
        return
    store = SharedStore(args.storage, args.reservation_minutes * 60)
    name = st.text_input(
        "Your name",
        help="Use the same name each time to resume drafts. Each person must use their own name.",
        key="annotator_name",
    )
    if store.is_review_snapshot:
        st.caption(
            "Offline review snapshot. Review as many images as you wish, then return this file."
        )
    batches = store.batches()
    has_model_review = any(b["dataset_role"] == "model_review" for b in batches)
    pages = (
        ("Review", "Progress and exports")
        if store.is_review_snapshot
        else (
            ("Annotate", "Review")
            + (("Model review",) if has_model_review else ())
            + ("Progress and exports", "Import batches")
        )
    )
    page = st.sidebar.radio("Workspace", pages)
    options = {
        b["batch_id"]: b["name"]
        for b in batches
        if (b["is_open"] or page == "Progress and exports")
        and (
            page in ("Progress and exports", "Import batches")
            or (b["dataset_role"] == "model_review") == (page == "Model review")
        )
    }
    batch_id = st.sidebar.selectbox(
        "Batch",
        [None, *options],
        format_func=lambda value: options[value] if value else "All batches",
    )
    try:
        if page in ("Annotate", "Review", "Model review"):
            if not name.strip():
                st.info("Enter your name to start or resume.")
                return
            if page == "Model review":
                model_review_queue(store, name, batch_id)
            else:
                with st.expander("Scoring rubric"):
                    st.write(
                        "Judge only visible content. Do not infer authenticity, origin, or hidden context."
                    )
                    st.write(
                        "Public relevance: 0 no evident public interest; 1 limited/contextual; 2 clear, broad public interest."
                    )
                    st.write(
                        "Harm urgency: 0 no observable urgent harm; 1 plausible/moderate concern; 2 severe or time-critical concern."
                    )
                    st.write(
                        "Vulnerability: 0 no observable cue; 1 a person or context visibly warrants added protection."
                    )
                    st.write(
                        "Sum: 0–1 low, 2–3 medium, 4–5 high. Rationale is optional; low confidence flags review by default."
                    )
                work_queue(store, name, batch_id, "annotation" if page == "Annotate" else "review")
        elif page == "Import batches":
            imports(store)
        else:
            st.dataframe(store.progress(), hide_index=True)
            st.caption(
                "Counts use unique image hashes. One image in multiple batches counts once in the overall total. Confirmed includes corrections approved by another person."
            )
            for kind, label in (
                ("current", "Current annotations"),
                ("confirmed", "Confirmed annotations"),
                ("history", "Review history"),
                ("model_review", "Model review decisions"),
            ):
                st.download_button(
                    label,
                    store.export_csv(kind, batch_id),
                    file_name=f"sensifake_{kind}.csv",
                    mime="text/csv",
                )
            st.caption(
                "Exports include saved work immediately, even for incomplete or closed batches. Each filename membership has a row; repeated hashes across batches are intentional. Provenance appears only in exports and import administration."
            )
    except (AnnotationError, OSError, ValueError) as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()

"""
Reusable Streamlit UI components for the experiment interface.

Provides display functinos for trials, commands, progress, and summaries.
"""

import streamlit as st

from models import Mode, RobotCommand, TrialDefinition


def render_header() -> None:
    """
    Component to render the application title and description.
    """

    st.title("Multimodal HRI Command Hub")
    st.caption("Voice and gesture input for structured robot commands.")


def render_mode_badge(mode: Mode) -> None:
    """
    Component to display the current interaction mode.

    Args:
        mode: Active input modality.
    """

    icons = {
        Mode.voice: "🔵",
        Mode.gesture: "🟢",
        Mode.multimodal: "🟣",
    }
    icon = icons.get(mode, "⚪")
    st.markdown(f"**Mode:** {icon} `{mode.value.upper()}`")


def render_trial_prompt(trial: TrialDefinition) -> None:
    """
    Component to display the current trial prompt

    Args:
        trial: Trial definition containing prompt text.
    """

    st.markdown("---")
    st.subheader(f"Trial {trial.trial_id}")
    st.info(trial.prompt_text)


def render_progress(current_index: int, total: int) -> None:
    """
    Component to display trial progress.

    Args:
        current_index: Index of the current trial.
        total: Total number of trials.
    """

    if total > 0:
        fraction = current_index / total
        st.progress(fraction, text=f"Trial {current_index + 1} of {total}")


def render_command_output(
    cmd: RobotCommand, correct: bool | None = None
) -> None:
    """
    Component to display recognised command with optional correctness.

    Args:
        cmd: Command to display.
        correct: True/False to show feedback, or None.
    """

    st.markdown("**Recognised Command:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Action", cmd.action.value if cmd.action else "—")
    col2.metric("Object", cmd.object.value if cmd.object else "—")
    col3.metric("Location", cmd.location.value if cmd.location else "—")
    col4.metric("Confidence", f"{cmd.confidence:.0%}")

    if correct is True:
        st.success("✓ Correct")
    elif correct is False:
        st.error("✗ Incorrect")


def render_command_panel(
    cmd: RobotCommand, correct: bool | None = None
) -> None:
    """
    Component to display command in a compact panel layout.

    Args:
        cmd: Command to display.
        correct: True/False for feedback, or None.
    """

    st.markdown("**Recognised Command**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Action", cmd.action.value if cmd.action else "—")
    c2.metric("Object", cmd.object.value if cmd.object else "—")
    c3.metric("Location", cmd.location.value if cmd.location else "—")
    st.caption(f"Confidence: {cmd.confidence:.0%}")

    if correct is True:
        st.success("✓ Correct")
    elif correct is False:
        st.error("✗ Incorrect")


def render_session_summary(summary: dict) -> None:
    """
    Component to display the end-of-session results.

    Args:
        summary: Dict returned by ExperimentRunner.get_summary().
    """

    st.subheader("Session Complete")
    st.write(f"**Participant:** `{summary['participant_id']}`")
    st.write(
        f"**Overall accuracy:** {summary['overall_accuracy']:.0%} "
        f"({summary['completed_trials']} / {summary['total_trials']} trials completed)"
    )

    if summary["by_condition"]:
        st.write("**Accuracy by condition:**")
        cols = st.columns(len(summary["by_condition"]))
        for col, (condition, stats) in zip(
            cols, summary["by_condition"].items()
        ):
            col.metric(
                condition.upper(),
                f"{stats['accuracy']:.0%}",
                f"{stats['correct']} / {stats['trials']} correct",
            )

# Experiment Design

## Overview

The experiment evaluates the effectiveness of multimodal interaction compared to unimodal approaches for structured robot command input.

Three interaction conditions are tested:

- Voice-only
- Gesture-only
- Multimodal (voice and gesture fusion)

The evaluation focuses on usability, accuracy, and efficiency when performing assistive pick-and-place tasks.

## Research Questions

The experiment is designed to address the following research questions:

### RQ1 - Performance

> Does multimodal interaction improve task performance compared to unimodal interaction?

Performance is measured using:
- Accuracy (correct command interpretation)
- Latency (time to complete input)
- Task completion efficiency (including corrections)

### RQ2 — User Experience

> How do users perceive the usability and intuitiveness of multimodal interaction compared to unimodal interaction?

Evaluation methods include:
- System Usability Scale (SUS)
- Post-task questionnaires
- Analysis based on usability principles (e.g. learnability, efficiency, error handling)

This focuses on interaction with the **system interface**, rather than the physical robot.

### RQ3 — Multimodal Interaction Behaviour

> What challenges arise when combining voice and gesture inputs in a shared interaction system?

This includes:
- Conflict between modalities
- Ambiguity in interpretation
- Temporal alignment issues
- System behaviour when inputs are incomplete or inconsistent

## Experimental Conditions

Each participant completes tasks under three conditions:

| Condition    | Description                            |
|--------------|----------------------------------------|
| Voice        | Commands issued using natural language |
| Gesture      | Commands issued using hand gestures    |
| Multimodal   | Combined voice and gesture input       |

All conditions use the same underlying task set.

## Task Design

Participants are asked to perform **structured pick-and-place tasks**.

Each task specifies:
- Action (e.g. pick, place, stop)
- Object (e.g. red cube)
- Location (e.g. left, right)

Tasks are presented as natural language prompts, for example:
```
"Pick up the red cube and place it on the left"
```
The system evaluates whether the interpreted command matches the expected outcome.

## Trial Structure

Each trial follows a consistent sequence:

1. **Prompt Display**
   - The participant is shown a task instruction

2. **Input Phase**
   - User provides input depending on condition:
     - Voice (speech or typed)
     - Gesture (webcam or buttons)
     - Multimodal (voice and gesture)

3. **System Interpretation**
   - The system parses and/or fuses the input into a structured command

4. **Feedback Phase**
   - The recognised command is displayed
   - The participant can:
     - Accept (correct)
     - Retry (counts as correction)

5. **Logging**
   - Trial result is recorded for analysis

![Trial Screenshot](/assets/trial-screenshot.png)
*Figure: Experiment mode interface showing prompt, input and result panel.*

## Trial Interaction Flow
The following sequence diagram illustrates the interaction flow during a trial in experiment mode, including user input, multimodal processing, fusion, and evaluation with logging.

![Experiment Mode Sequence Diagram](/assets/exp-sequence-diagram.png)


### Counterbalancing

To reduce ordering effects, the experiment uses **counterbalanced condition ordering**.

Participants are assigned to one of three condition sequences:

- Voice $\to$ Gesture $\to$ Multimodal
- Gesture $\to$ Multimodal $\to$ Voice
- Multimodal $\to$ Voice $\to$ Gesture

Assignment is determined using a deterministic mapping based on participant ID.

This ensures:
- Each condition appears equally in each position
- Learning effects are distributed across conditions

### Number of Trials

Each condition includes: `TRIALS_PER_CONDITION = 10`

Total trials per participant: 3 conditions $\times$ 10 trials = 30 trials

---

## Metrics Collected

The system records the following metrics for each trial:

| Metric                | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| **Accuracy**          | Whether the predicted command matches the expected command       |
| **Latency**           | Time taken from prompt to final accepted command ($\text{ms}$)   |
| **Correction Count**  | Number of retries before correct submission                      |
| **Conflict Rate**     | Frequency of disagreement between voice and gesture inputs       |
| **Fusion Timing**     | Whether inputs occurred within the fusion time window            |
| **Confidence**        | Final confidence score of the predicted command                  |


## Data Logging

All trial data is stored in CSV format for reproducibility and analysis.

Each row contains:
- Participant ID
- Condition
- Trial ID
- Expected vs predicted fields
- Accuracy
- Latency
- Correction count
- Fusion metadata (timestamps, conflicts, etc.)
- Confidence score

Logs are saved to: `\logs` which is created automatically at runtime.

---
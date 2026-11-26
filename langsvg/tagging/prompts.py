"""Prompt strings used by Lang-SVG tagging, kept verbatim so results reproduce.

* ``OPEN_WORLD_SYSTEM_PROMPT`` / ``NODE_TAGGING_INSTRUCTION``: open-vocabulary hierarchical tagging.
* ``PASCAL_SYSTEM_PROMPT`` / ``PASCAL_NODE_INSTRUCTION``: open-vocabulary tagging with the
  PascalPart116 object categories as a soft prior on object identity.
* ``GENERAL_VERIFICATION_SYSTEM_PROMPT`` / ``VERIFICATION_INSTRUCTION``: optional second pass
  that keeps or revises a first-pass prediction.
* ``CLOSED_SET_SYSTEM_PROMPT``: closed-set (pick one candidate label) classification, the exact
  runtime prompt of the ADE20KPart234 hidden-tag recovery benchmark (setting A4). The paper
  appendix prints it without the trailing JSON output schema.
"""

PASCAL_OBJECT_CATEGORIES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

# The two complete prompts share all common rules, examples, and response text.
# Pascal changes the introduction and adds rules within the same Rules section.
OPEN_WORLD_INTRO = """You are an open-world assistant for semantic tagging in an SVG tree.
Use the full image, target crop, bounding box, and current hierarchy context.
"""

COMMON_TAGGING_RULES = """- Identify the target using visual evidence, spatial layout, and world knowledge.
- Use ancestor, parent, sibling, and child tags as context; let clear visual
  evidence override an incorrect contextual label.
- Choose object for a standalone entity, part for an object component, and
  subpart for a finer structural component. Prefer the most specific supported
  label and preserve meaningful object-part relations in its hierarchical tag.
- Treat candidate labels as soft priors, not an exhaustive vocabulary. Apply
  each candidate list only to the semantic level it describes.
- Prefer a candidate when it fits. If strong visual evidence supports none of
  the relevant candidates, predict the appropriate open-world label.
- Use texture only for appearance-only detail. Use unknown only when visual
  evidence and hierarchy do not support a stable semantic interpretation.
"""

TAGGING_EXAMPLES = """Examples:
- ancestor [person] and target around head -> {"category":"part","tag":"person's head"}
- ancestor [person's head] and target around eye -> {"category":"subpart","tag":"person's head's eye"}
- ancestor [person's head's eye] and small dark round target -> {"category":"subpart","tag":"person's head's eye's pupil"}
- ancestor [person's head's eye] and target is only a color/glare patch -> {"category":"texture","tag":"person's head's eye's texture"}
- ancestor [car] and lower round region -> {"category":"part","tag":"car's wheel"}
- ancestor [car's wheel] and black outer ring -> {"category":"subpart","tag":"car's wheel's tire"}
"""

TAGGING_RESPONSE_INSTRUCTION = """Return the final semantic category and a concise hierarchical tag.
"""

PASCAL_INTRO = """You are a visual assistant for semantic tagging in PascalPart116 SVG trees.
Use the full image, target crop, bounding box, and current hierarchy context.
Treat the following PascalPart116 object categories as soft priors for object identity:

aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable,
dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.
"""

PASCAL_ADDITIONAL_RULES = """- Apply this list only to object identity, including the containing object's
  identity when the target is a part or subpart. Prefer a listed object category
  when it matches the visual evidence.
- If strong visual evidence supports an object outside the list, predict that
  open-world category rather than forcing a listed category.
- Label parts and subparts freely within their object context. A wheel or eye
  is not required to belong to this object-category list. Do not use texture or
  unknown merely because a supported semantic label is absent from the list.
"""

OPEN_WORLD_SYSTEM_PROMPT = (
    OPEN_WORLD_INTRO + "\nRules:\n" + COMMON_TAGGING_RULES
    + "\n" + TAGGING_EXAMPLES + "\n" + TAGGING_RESPONSE_INSTRUCTION
)

PASCAL_SYSTEM_PROMPT = (
    PASCAL_INTRO + "\nRules:\n" + COMMON_TAGGING_RULES
    + PASCAL_ADDITIONAL_RULES + "\n" + TAGGING_EXAMPLES
    + "\n" + TAGGING_RESPONSE_INSTRUCTION
)

GENERAL_VERIFICATION_SYSTEM_PROMPT = """You verify semantic tags for nodes in a visual SVG hierarchy.
Review the proposed tag using the full image, target crop, bounding box, and
current ancestor, parent, sibling, and child context.

Rules:
- Check that the tag matches the visible region, its spatial position, and the
  object-part relations in the tree, with an appropriate semantic level.
- Keep a correct prediction. Revise an implausible, overly broad, or
  hierarchy-inconsistent prediction to the most specific supported label.
- Treat candidates as soft priors at their stated semantic level. An object
  category list applies to object identity, not to part or subpart labels.
- Retain or predict an out-of-list label when strong visual evidence supports
  it. Do not force a listed category merely to satisfy a candidate vocabulary.
- Use texture only for appearance-only detail. Use unknown only when visual
  evidence and hierarchy do not support a stable semantic interpretation.

Return the final semantic category and tag. Indicate whether the prediction was
kept or revised, and give a brief verification reason.
"""

NODE_TAGGING_INSTRUCTION = (
    "Predict the most specific supported semantic category and hierarchical tag "
    "for this target. Use the image, crop, geometry, and current tree context. "
    "Candidate labels are soft priors at their stated semantic level; predict "
    "an out-of-list label when strong visual evidence supports it. "
    "Return only category and tag."
)

PASCAL_NODE_INSTRUCTION = NODE_TAGGING_INSTRUCTION + (
    " The Pascal object categories are a soft prior for object identity only. "
    "For a part or subpart, use its containing object as context and predict "
    "the part/subpart label freely."
)

VERIFICATION_INSTRUCTION = (
    "Verify the first-pass prediction against visual evidence and tree context. "
    "Keep a supported tag or revise an incorrect one. Apply each candidate list "
    "only at its stated semantic level, and retain a supported out-of-list "
    "prediction. Return the final category and tag, a keep/revise decision, "
    "and a brief reason."
)


CLOSED_SET_SYSTEM_PROMPT = """You are an ADE20KPart234 scene-graph part classification evaluator.

You will receive:
- the original full image
- a crop around the containing object, when available
- a crop around the target bbox, when available
- metadata copied from a layered scene-graph JSON
- known object/part context and sibling layout from that JSON
- bbox layout metadata for the target, parent, siblings, and parent children
- a closed candidate list of valid part/subpart labels

The ground-truth target node tag has been hidden. Classify the semantic part or subpart
inside the target bbox using the image and the unmasked scene-graph context.

Rules:
- Choose exactly one label from candidate_labels.
- Use the target bbox metadata to locate the masked part/subpart inside the full image and object crop.
- Use the object crop for object-level context, not as the target region itself.
- Use the target bbox crop for local visual details.
- Use ancestor_tags, parent_tag, sibling_tags, child_tags, parent_layout,
  sibling_bboxes, and parent_child_layout from the JSON as context.
- candidate_labels are the only valid outputs. They are full hierarchical labels;
  reason about their leaf part names, but output the exact candidate label string.
- predicted_label must exactly match one candidate label string.
- Do not invent a new label outside the candidate list.
- Return exactly one JSON object.
- Do not output prose, markdown, or reasoning text.

Before choosing the label, silently run this parent-to-subpart checklist:
1. Identify the parent object/part from ancestor_tags, parent_tag, and parent_layout.
2. Generate plausible child parts for that parent, then eliminate candidates whose
   position, size, color/material, or sibling layout is inconsistent.
3. Compare the target bbox with every sibling bbox under the same parent. Existing
   known siblings are strong constraints; masked siblings may still provide layout
   evidence through their bbox and color but not their hidden tag.
4. Use bbox_average_rgb and visual color/material continuity as supporting evidence.
   Similar color/texture can indicate another segment of the same object; very
   different color may indicate clothing, shoe, material detail, or texture.
5. Avoid impossible duplicate anatomy or object parts unless the image and layout
   clearly support duplication. Choose "unknown" only if no candidate label is
   spatially and visually plausible, but prefer a valid candidate when evidence is
   sufficient.

Examples of the intended reasoning:
- If parent is person and known siblings already include left/right arm, a third
  arm candidate at the lower body is usually wrong; a lower bbox near feet is more
  likely leg, foot, shoe, or clothing depending on shape and color.
- If parent is person's leg and the target bbox is at the bottom end with a shoe-like
  color/shape, prefer the candidate whose leaf is foot or shoe over hand or arm.
- If parent is person's face/head, small upper symmetric boxes are likely eyes; a
  central box below eyes and above mouth is likely nose; a lower face box is more
  likely mouth/chin than eye.
- If parent is car/vehicle, lower left/right boxes are likely wheel/tire; boxes in
  the upper body are more likely window, door, mirror, or light.
- If one known sibling is a wheel on the lower left and the target is a similar
  colored/size lower-right box, the target is likely another wheel candidate.

Output schema:
{
  "predicted_label": "object's part",
  "confidence": 0.0,
  "reason": "brief visual reason"
}
"""

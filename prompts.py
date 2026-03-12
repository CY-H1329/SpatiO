from typing import Dict, List, Optional


def build_head_agent_prompt(
    query: str,
    category_list: List[str],
    category_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    return f"""You are the Head Agent of SpatiO. Classify the spatial query into exactly ONE category
below, based on the primary geometric predicate the question asks to resolve.

Internally analyse before answering (do NOT output this):
  (1) Primary spatial predicate?  e.g. vertical stacking | lateral adjacency | depth | cardinality
  (2) Multiple predicates present? If so, which is the main ask?
  (3) Single object, or relationship between multiple objects?
  (4) Reference frame: camera/viewer, or another scene object?

## CATEGORIES
  spatial_relation   WHERE one object is relative to another in 3D space: above/below,
                     next to, between, on top of, at a higher elevation.
                     e.g. "Is the clock above the bus?"  "Is the bench beside the tree?"
  distance_depth     HOW FAR from the camera/viewer or a third reference object.
                     e.g. "Which is closer to the camera?"  "Which person is farther?"
  size               HOW BIG -- comparing physical scale, dimension, or extent.
                     e.g. "Is the truck larger than the car?"  "Which building is taller?"
  orientation        WHICH WAY -- facing direction, left/right, front/behind,
                     parallel or perpendicular arrangement relative to viewpoint.
                     e.g. "Is the lamp to the left of the sofa?"
  counting           HOW MANY -- cardinality of a set of objects in the scene.
                     e.g. "How many people are near the entrance?"

## RULES
  1. HOW MANY / COUNT  ->  counting  (regardless of any spatial context in the question)
  2. Depth / proximity from camera  ->  distance_depth  (NOT spatial_relation)
  3. "taller" (physical size) -> size  |  "higher" (3D elevation) -> spatial_relation
  4. Facing direction, left/right, front/behind, angular arrangement  ->  orientation
  5. Classify by the predicate DIRECTLY ASKED, not by surrounding spatial context clauses.

## DO NOT  |  Output reasoning  |  Output anything except the category name  |  Invent categories

## INPUT / OUTPUT
  Question : {query}
  Output   : Respond with ONLY the category name. Nothing else."""


_ROLE1_PROMPT = """You answer spatial reasoning questions using pictorial depth cues -- occlusion,
relative size, height in image, familiar size -- WITHOUT constructing explicit 3D models.
Always anchor reasoning to a reference object.

Internally select protocol (do NOT output):  HOW MANY/COUNT -> COUNT PROTOCOL  |  else -> SPATIAL PROTOCOL

## PICTORIAL CUES  (apply in priority order)
  occlusion          A hides B -> A is closer          e.g. A partially covers B -> A is in front
  relative_size      Larger apparent size -> closer     e.g. A looks larger than same-size B -> A nearer
  height_in_image    Lower in frame -> closer           e.g. near bottom edge -> foreground
  familiar_size      Use known real-world size          e.g. very small car -> far away

## COUNT PROTOCOL  (HOW MANY / COUNT)
  1. Unit definition   ONE instance first. Multiple parts of same object = 1.  e.g. train = loco + cars
  2. Systematic scan   Top-left -> centre -> bottom-right -> edges. Nothing skipped.
  3. Occlusion rule    Partially visible distinct instance = 1.  Same-object parts NOT each 1.
  4. Semantic match    Broad definitions.  e.g. countertop as table -> include
  5. Re-check          List each instance. Avoid double-counting.

## SPATIAL PROTOCOL  (position / depth / distance / orientation)
  Step 1  Decompose    Atomic sub-questions.  e.g. "Where is X rel. Y?" -> (a) X?  (b) Y?  (c) relation?
  Step 2  Anchor       Reference object; describe its image position.  e.g. upper-left, centre-frame
  Step 3  Cue+Resolve  Apply cues in order; state which cue supports conclusion.
                       Priority: occlusion > relative_size > height_in_image > familiar_size

## DO NOT  |  Infer 3D coordinates  |  Output protocol selection  |  Exceed 150 words in justification

## INPUT / OUTPUT
  Question : {query}
  Output   : Answer first, then justification <= 150 words.
             Format:  <Answer>  /  Reason: <justification>"""


_ROLE2_PROMPT = """You answer spatial reasoning questions by interpreting structured 3D representations
computed from the input image. Do NOT rely on pictorial heuristics -- trust the tool outputs.

Internally select protocol (do NOT output):  HOW MANY/COUNT -> COUNT PROTOCOL  |  else -> 3D PROTOCOL

## TOOL SUITE
  depth_map       Dense per-pixel metric depth (DepthPro). Normalised z in [0,1], 0=closest.
  instance_mask   Per-object pixel mask and centroid (SAM2). 2D location, bbox, mask area.
  point_cloud     3D centroids lifted from depth+mask (SpatialVLM). Centroid (X,Y,Z) + extent.
  surface_normal  Per-region normal vectors. Indicates planarity and facing direction.
  bbox_3d         3D bounding box: centroid, dims (WxHxD m), orientation angle.

## COUNT PROTOCOL  (HOW MANY / COUNT)
  1. Unit definition    ONE instance. Cross-reference instance_mask with Tool 9 count.
  2. Mask validation    Each SAM2 mask = one candidate. Reject below-threshold area masks.
  3. Depth separation   dz < 0.05 between two masks -> verify not same object.
  4. Semantic match     bbox_3d dims fit category -> include. Use broad definitions.
  5. Re-check           Cross-validate Tool 9 against mask count. Report discrepancies.

## 3D PROTOCOL  (depth / position / distance / orientation / size)
  Step 1  Decompose      Atomic sub-questions.  e.g. "Is A above B?" -> Y_A? | Y_B? | Y_A > Y_B?
  Step 2  Select tools   depth ordering / proximity -> Tools 3,4  |  3D position / elevation -> Tool 5
                         inter-object distance -> Tool 8  |  size comparison -> Tool 7  |  orientation -> Tools 6,7
  Step 3  Resolve        State numerical evidence; cite supporting tool.
                         Trust: point_cloud > depth_map > surface_normal > instance_mask
                         Flag ambiguous cases (dz < 0.05, missing normals) explicitly.

## DO NOT  |  Use pictorial heuristics when tools available  |  Hallucinate tool values  |  Exceed 150 words

## INPUT / OUTPUT
  Question : {query}
  Output   : Answer first, then justification <= 150 words citing specific tool values.
             Format:  <Answer>  /  Reason: <justification with tool references>"""


_ROLE3_PROMPT = """You answer spatial reasoning questions by combining three inputs:
  (1) Image       Visual context -- layout, occlusion, appearance.
  (2) Query       Objects, relations, answer options.
  (3) Scene graph Structured nodes + edges from the image  (JSON; see tool figure).

Graph is PRIMARY structured source. Always cross-check with the image.
If graph contradicts image or is incomplete -> prioritise image; state "Graph inconsistency detected."

Internally select protocol (do NOT output):  HOW MANY/COUNT -> COUNT PROTOCOL  |  else -> GRAPH PROTOCOL

## COUNT PROTOCOL  (HOW MANY / COUNT)
  1. Node enumeration   Count matching-label nodes; cross-validate with image.
  2. Confidence filter  Exclude nodes with score < 0.5.
  3. Occlusion check    Visible in image but absent from graph -> add if clearly distinct.
  4. Semantic match     Broad labels.  e.g. "worksurface" -> "table"
  5. Re-check           Graph count vs image scan. Report discrepancies.

## GRAPH PROTOCOL  (relation / position / orientation / depth)
  Step 1  Parse query    (a) subject  (b) reference object  (c) relation type
                         e.g. "Is A left of B?" -> subject=A, ref=B, rel=left_of
  Step 2  Locate nodes   Label match; multiple same-label -> pick highest score.
  Step 3  Traverse       Direct:  edge(subject=A, relation=R, object=B) -> confirms R(A,B)
                         Inverse: edge(subject=B, relation=R', object=A) -> confirms R(A,B)
                         Pairs:   above<->below  |  left_of<->right_of  |  in_front_of<->behind
                         Also:    overlaps (symmetric)  |  closer_to / farther_from
  Step 4  Cross-check    Verify result vs image layout. If contradicted -> trust image.
  Step 5  Map to option  Answer node label -> correct option (A)/(B)/(C)/(D).

## FALLBACK  |  Empty/failed -> image only; "Tool unavailable."
              |  Edge missing -> check inverse; absent -> use image.
              |  Label ambiguous -> highest-score node; note ambiguity.
              |  Graph contradicts image -> trust image; "Graph inconsistency detected."

## DO NOT  |  Rely on graph alone when image contradicts  |  Hallucinate JSON values  |  Exceed 150 words

## INPUT / OUTPUT
  Question : {query}
  Output   : Answer first, then justification <= 150 words citing edges and node IDs.
             Format:  <Answer>  /  Reason: <justification with graph references>"""


_ROLE_PROMPTS = {
    "direct_visual_heuristic": _ROLE1_PROMPT,
    "explicit_3d_representation": _ROLE2_PROMPT,
    "scene_graph_construction": _ROLE3_PROMPT,
}


def _get_output_format_specialist(answer_type: str) -> str:
    if answer_type == "free_form":
        return "Answer: <value>  e.g. 3, two, red, left  |  Reason: <justification <= 150 words>"
    return "Answer: (A) or (B) or (C) or (D) or (E) or (F)  |  Reason: <justification <= 150 words>"


def build_role_prompt(
    role: str,
    query: str,
    tool_output: Optional[str] = None,
    answer_type: str = "multiple_choice",
) -> str:
    template = _ROLE_PROMPTS.get(role)
    if template is None:
        raise ValueError(f"Unknown role: {role!r}")

    prompt = template.format(query=query)

    if tool_output and role in ("explicit_3d_representation", "scene_graph_construction"):
        prompt = prompt + "\n\n## TOOL OUTPUT (use this data in your reasoning)\n\n" + tool_output

    return prompt


ROLE2_TOOL_TEMPLATE = """## Tool 1.  Depth Map Grid  (3x3, DepthPro, normalised; 0=closest, 1=farthest)
  top-left:{tl}    top-center:{tc}    top-right:{tr}
  mid-left:{ml}    center:{cc}        mid-right:{mr}
  bot-left:{bl}    bot-center:{bc}    bot-right:{br}

## Tool 2.  Instance Masks & 2D Centroids  (SAM2)
  {obj_1}: centroid=({cx},{cy}px)  bbox=[{x1},{y1},{x2},{y2}]  area={area}px2
  {obj_2}: centroid=({cx},{cy}px)  bbox=[{x1},{y1},{x2},{y2}]  area={area}px2

## Tool 3.  Object Depth Values  (DepthPro, normalised z, 0=closest)
  1. {obj_1} ({region}): z={val}  [CLOSEST]     2. {obj_2}: z={val}     3. {obj_3}: z={val}  [FARTHEST]

## Tool 4.  Depth Ordering  (z_A < z_B => A in front of B)
  {obj_A} (z={za}) -> {obj_B} (z={zb})  [dz={delta}]

## Tool 5.  3D Point Cloud Centroids & Extents  (camera space, metres)
  {obj_1}: centroid=(X={x}, Y={y}, Z={z}m)  extent=(W={w}, H={h}, D={d}m)
  {obj_2}: centroid=(X={x}, Y={y}, Z={z}m)  extent=(W={w}, H={h}, D={d}m)

## Tool 6.  Surface Normals          {obj_1}: normal=({nx},{ny},{nz})  facing={front/up/left/right/away}
## Tool 7.  3D Bounding Boxes        {obj_1}: centroid=(X,Y,Z)  dims=(WxHxD m)  orientation={angle}deg
## Tool 8.  Inter-Object Distances   {obj_A} <-> {obj_B}: d={dist}m  (dX={dx}, dY={dy}, dZ={dz})
## Tool 9.  Instance Count           {object_type}: {count}

## Interpretation Guide
  z (norm.)   0=closest, 1=farthest.  dz < 0.05 -> ambiguous; verify with Tool 5.
  Z (metric)  Depth metres. More reliable than normalised z for distance queries.
  normal      dot(n,(0,0,-1)) > 0 -> facing camera. Use for orientation queries.
  dims        WxHxD metres. Use for size comparison.
  dist (T8)   3D Euclidean metres. Use for proximity queries.
  Trust       point_cloud > depth_map > surface_normal > instance_mask > pictorial"""


ROLE3_TOOL_TEMPLATE = """## Scene Graph (JSON)
{
  "nodes": [
    {"id":"1", "label":"{obj_1}", "bbox":[{x1},{y1},{x2},{y2}], "center":[{cx},{cy}], "score":{conf}},
    {"id":"2", "label":"{obj_2}", "bbox":[{x1},{y1},{x2},{y2}], "center":[{cx},{cy}], "score":{conf}},
    {"id":"3", "label":"{obj_3}", "bbox":[{x1},{y1},{x2},{y2}], "center":[{cx},{cy}], "score":{conf}}
  ],
  "edges": [
    {"subject":"{id}", "relation":"{rel}", "object":"{id}"},
    {"subject":"{id}", "relation":"{rel}", "object":"{id}"}
  ]
}

## Node Schema
  id      Unique integer string identifier.
  label   Object category (COCO-style). Match to query objects by label.
  bbox    [x1,y1,x2,y2] pixel coordinates. Use for spatial layout verification.
  center  [cx,cy] centroid pixels. Confirm left/right, above/below.
  score   Detection confidence [0,1]. score < 0.5 -> exclude.

## Edge Schema
  subject/object   Node IDs.  subject -> relation -> object.
  relation         above | below | left_of | right_of | in_front_of | behind | overlaps | closer_to | farther_from

## Traversal Guide
  "Is A above B?"       edge(subj=A_id, rel=above,        obj=B_id) ?
  "What is left of X?"  edge(rel=left_of,   obj=X_id) -> subject is answer
  "What is above X?"    edge(rel=above,      obj=X_id) -> subject is answer
  "Is A in front of B?" edge(subj=A_id, rel=in_front_of, obj=B_id) ?
  "Which is closer?"    edge(rel=closer_to)             -> subject is nearer
  Inverse check: above<->below  |  left_of<->right_of  |  in_front_of<->behind

## Consistency Check  (run before accepting any result)
  above/below        center_y(subject) < center_y(object)   [Y: 0=top]
  left_of/right_of   center_x(subject) < center_x(object)   [X: 0=left]
  Coordinates contradict edge -> flag; reason from image."""


def build_final_reasoning_prompt(
    query: str,
    shared_memory_text: str,
    category: str,
    answer_type: str,
    with_image: bool = False,
) -> str:
    answer_type_str = "multiple_choice" if answer_type == "multiple_choice" else "open_ended"
    image_note = (
        "\n\nYou also see the image that the specialists analysed. Cross-check their reasoning against what you observe."
    ) if with_image else ""

    output_format = """## OUTPUT FORMAT  (STRICT)
  multiple_choice   Answer: (A) or (B) or (C) or (D) or (E) or (F)
                    Reason: <2-5 sentences: question | agent selection | synthesis | conclusion>
  open_ended        Answer: <value>   e.g.  2.4  |  3  |  "to the left of the sofa"
                    Reason: <2-5 sentences: computation | agents | aggregation | confidence>"""

    return f"""You are the Final Reasoning Agent of SpatiO. Three specialists have independently analysed the
same image and question. You receive their outputs and a TTO-derived weight w per agent.
w in [0,1] = observed accuracy for this task category + role, accumulated over the query stream.
Use w to MODULATE -- not replace -- your reasoning.{image_note}

## INPUTS
  Question    : {query}
  Category    : {category}
  Answer type : {answer_type_str}

{shared_memory_text}

## STEP 1 -- ANSWER TYPE
  multiple_choice   Options (A)/(B)/... given. Pick best match. Do NOT go outside the option set.
  open_ended        Compute from specialist evidence:
    float    Weighted avg over role-relevant agents: answer = sum(w_i * val_i) / sum(w_i)
    int      Weighted plurality; round to nearest integer.
    string   Synthesise from highest-w, most-relevant agent's output.

## STEP 2 -- EVALUATE AGENTS  (for each agent)
  Is the role appropriate?  depth -> explicit_3d  |  2D relation -> scene_graph  |  layout -> implicit_visual
  High w + relevant role -> strong signal.   Low w -> weak evidence regardless of answer.
  Internally consistent? Did the agent use its own data correctly?
  High-w agent with IRRELEVANT role: down-weight by relevance.

## STEP 3 -- WEIGHTED SYNTHESIS
  Agreement among high-w, role-relevant agents -> strong evidence.
  Disagreement -> favour role-matched agent citing most concrete data (z values, graph edges, cues).
  Do NOT follow majority vote blindly. One high-w relevant agent can override two weaker ones.
  |w_i - w_j| < 0.1 for all pairs -> reason from evidence content alone.
  open_ended float + diverging values -> weighted average; flag "+/- range".

## STEP 4 -- CONCLUSION
  Most justified answer (synthesis + answer type). Cite convincing agents: role, w, evidence.
  Inconclusive -> best-supported answer; state uncertainty explicitly.

## WEIGHT INTERPRETATION
  w >= 0.75          High reliability. Strong prior for this category + role.
  w in [0.50, 0.75)  Moderate. Consider alongside reasoning quality.
  w < 0.50           Low. Require strong evidence to trust.
  w = 0.50           Unobserved prior. Reason from content alone.

## ROLE-RELEVANCE PRIORITY
  spatial_relation / size   scene_graph > implicit_visual > explicit_3d
  distance_depth            explicit_3d > implicit_visual > scene_graph
  orientation               explicit_3d > scene_graph     > implicit_visual
  counting                  scene_graph > explicit_3d     > implicit_visual

## DO NOT  |  Fixed majority vote  |  Override high-w relevant agent without justification
           |  Value outside option set (multiple_choice)  |  Hallucinate values (open_ended)
           |  Exceed 5 sentences in Reason  |  Output anything before "Answer:"

{output_format}
"""

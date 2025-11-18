import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List, Optional
import json, re, ollama, time

# ─────────────────────────────────────────────────────────────
# 데이터 구조: 월드 모델(JSON)
# ─────────────────────────────────────────────────────────────
@dataclass
class ObjNode:
    id: str                         # SLAM에서 넘어온 고유 ID("cup#3")
    klass: str                      # 클래스명("cup")
    map_xy: Tuple[float, float]     # 맵 좌표계 절대좌표(x, y)

@dataclass
class WorldModel:
    objects: Dict[str, ObjNode] = field(default_factory=dict)

    def apply_slam_json(self, slam_json: str | dict) -> int:
        data = json.loads(slam_json) if isinstance(slam_json, str) else slam_json
        objs = data.get("objects", data) if isinstance(data, dict) else data

        updated = 0
        for n in objs:
            oid = str(n["id"])
            klass = str(n["class"])
            mx, my = n["map_xy"]
            self.objects[oid] = ObjNode(
                id=oid,
                klass=klass,
                map_xy=(float(mx), float(my)),
            )
            updated += 1
        return updated

    def to_json(self, indent: int | None = 2, ensure_ascii: bool = True) -> str:
        # 항상 id 오름차순 정렬
        doc = {
            "objects": [
                {"id": oid, "class": node.klass, "map_xy": [node.map_xy[0], node.map_xy[1]]}
                for oid, node in sorted(self.objects.items(), key=lambda kv: kv[0])
            ]
        }
        return json.dumps(doc, ensure_ascii=ensure_ascii, indent=indent)
    
# ─────────────────────────────────────────────────────────────
# 인벤토리 모델 (NEW)
# ─────────────────────────────────────────────────────────────
@dataclass
class InventoryModel:
    items: Dict[str, ObjNode] = field(default_factory=dict)  # key = id

    def to_json(self, indent: int | None = 2, ensure_ascii: bool = True) -> str:
        doc = {
            "holding": [
                {"id": oid, "class": node.klass}
                for oid, node in sorted(self.items.items(), key=lambda kv: kv[0])
            ]
        }
        return json.dumps(doc, ensure_ascii=ensure_ascii, indent=indent)

    def to_dict(self) -> dict:
        return json.loads(self.to_json(indent=None))

    def add_from_world(self, world: "WorldModel", ref: str) -> bool:
        # ref가 id면 id 우선, 아니면 class 기반 단일 매칭 시 추가
        if ref in world.objects:
            self.items[ref] = world.objects[ref]
            return True
        # class 이름으로 유일 매칭 시
        matches = [n for n in world.objects.values() if n.klass == ref]
        if len(matches) == 1:
            self.items[matches[0].id] = matches[0]
            return True
        return False

    def remove_by_ref(self, ref: str) -> bool:
        # id 제거 우선
        if ref in self.items:
            del self.items[ref]
            return True
        # class 이름으로 유일 매칭 시
        matches = [oid for oid, node in self.items.items() if node.klass == ref]
        if len(matches) == 1:
            del self.items[matches[0]]
            return True
        return False

# ─────────────────────────────────────────────────────────────
# LLM 백엔드
# ─────────────────────────────────────────────────────────────
Actor_Prompt = """
You are a helpful planner for a robot.

The control runs in iterations:
- user_instruction: a natural-language task.
- world_json: the current world state with this exact minimal schema:
  {
    "objects": [
      { "id": "<string>", "class": "<string>", "map_xy": [<float x>, <float y>] },
      ...
    ]
  }
- inventory_json: the robot's currently held objects:
  {
    "holding": [
      { "id": "<string>", "class": "<string>" }
    ]
  }
- id: stable unique ID from SLAM (e.g., "cup#3"); do not invent new IDs.
- class: semantic label from the detector (lowercase singular, e.g., "cup").
- map_xy: absolute [x, y] in the map frame; do not output or alter coordinates.
- history_pseudo_function: a chronological log of the function calls the robot has actually executed so far. use this as context when generating the plan.

YOUR JOB:
- Plan a short sequence of high-level function calls that drives progress toward the user instruction.
- If the required target(s) are not visible in world_json, call search() to find target.
- When the mission is complete, call return_to_base() and stop.

ALLOWED FUNCTIONS:
- move_to(object): Move toward the specified object or location.
- search(object): When there is no relevant object for the mission in world_json, it explores unexplored areas to find such an object.
- pick(object): Pick up the specified object.
- place(object): Put down the held object at the current location.
- return_to_base(): Return to the robot's base location.
- speak(message):  Say the given message aloud using TTS.

HARD RULES:
- List function calls only. Do not use control flow (if/else, for/while), variable definitions, imports, or any other Python syntax.
- Output nothing after search() in the same plan.
- If an object required to complete the task is not present in the current world_json, call search().
- Do not output coordinates; reference objects by id or class only.
- Use history_pseudo_function to avoid repeating already executed steps unless repetition is necessary to make progress.
- If an object appears in inventory_json.holding, treat it as currently held even if it is absent from world_json; do NOT call pick() for it. Prefer place() when appropriate.

OUTPUT FORMAT (STRICT):
1) Reasoning (plain text, 1-5 short bullet lines explaining the plan)
2) Pseudocode (exactly one fenced python block containing only pseudo_function, one per line)
"""

Critic_Prompt = """
You are a generous robot plan critic.
Your task is to evaluate the pseudo_function plans proposed by the planner.
If the score is insufficient, analyze the surrounding context and recommend the necessary steps and processes.

ALLOWED FUNCTIONS:
- move_to(object): Move toward the specified object or location.
- search(object): Use when an object required to complete the mission is not visible. As an exception, search(object) may take an object that is not currently visible in the world_json as its argument.
- pick(object): Pick up the specified object.
- place(object): Put down the held object at the current location.
- return_to_base(): Return to the robot's base location.
- speak(message):  Say the given message aloud using TTS.

Pseudo_function must be composed using only the allowed functions above.

If an object required to complete the task is not present in the current world_json, call search().
Do not include any functions after search() (such as move_to, pick, or place).
Any actions listed after search() would be based on outdated assumptions and may no longer be valid.
After search() is executed, the environment may change based on the new world_json.

You must ensure that, except for search(object), the plan uses only the target objects that are present in the world_json.
Using objects not present in the current world_json is a critical flaw, **except when they appear as the argument to search(object)**. This exception is intentional and must not be penalized.

When certain functions like pick() are used, the corresponding object may be removed from the world_json in subsequent iterations.
This does not mean the object is unknown — it simply reflects that the robot is already holding it.
You must take this into account by referring to the *history_pseudo_function* when determining whether an object should be considered known or available.
Also consult *inventory_json*; if an object is listed there, it is currently held and should not be re-picked.

Review the plan using the following 3 criteria:
1. Logical correctness — Are the steps valid and coherent?
2. Context awareness — Does the plan correctly respond to present or known elements in the world_json?
3. Completeness — Does the plan address the user instruction as much as possible this iteration?

For each criterion, return:
- A 1-line comment
- A score from 1 to 5 (5 = perfect)
- If the pseudo_function is not reasonable, explain why and recommend the necessary steps.

Finally, return a verdict line at the end:
- VERDICT: ACCEPTED
- VERDICT: REVISE REQUIRED

Only mark as REVISE REQUIRED if:
- The plan uses forbidden constructs (e.g., if/else)
- It applies when search() is used two or more times.
- It ignores present or reachable goals in this iteration.
- It contains any invalid or undefined function

Do NOT require revision for minor stylistic issues or suboptimal naming.

Format your output exactly as:
Logical correctness: [comment] (Score: x)
Context awareness: [comment] (Score: x)
Completeness: [comment] (Score: x)
Overall assessment of the plan: [comment]
VERDICT: ACCEPTED or REVISE REQUIRED

If the total score (Logical correctness + Context awareness + Completeness) exceeds **10**, set VERDICT: ACCEPTED; otherwise set VERDICT: REVISE REQUIRED.
"""

PLANNER_MODEL = "qwen2.5vl:7b"   # VLM (planner 용)
CRITIC_MODEL  = "qwen2:7b"     # LLM (critic 용)

TEMPERATURE_PLANNER = 0.2
TEMPERATURE_CRITIC  = 0.2

def call_ollama(messages, model=str, temperature: float = 0.2):
    resp = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return resp["message"]["content"]

def extract_pseudo_function(text: str) -> list[str] | None:
    m = re.search(r"```(?:python)?\s*(.*?)""`", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()
    return None

def planner(
    Actor_Prompt: str,
    user_instruction: str,
    world_json: dict,
    history_pseudo_function: list[str] | None = None,
    critic_feedbacks: str | None = None,
    inventory_json: dict | None = None,
):
    
    if history_pseudo_function:
        loop_history = "\n\n".join(
            f"Loop {i}\n```python\n{code.strip()}\n```"
            for i, code in enumerate(history_pseudo_function, 1)
        )
    else:
        loop_history = "[None yet]"
    
    if critic_feedbacks:
        fb = critic_feedbacks.strip()
        feedback_history = f"Feedback\n{fb}"
    else:
        feedback_history = "[None yet]"

    inv_block = json.dumps(inventory_json or {"holding":[]}, ensure_ascii=True, indent=2)
    
    messages = [
        {"role": "system", "content": Actor_Prompt},
        {"role": "user",
         "content":
            "User instruction:\n"
            f"{user_instruction}\n\n"
            "world_json:\n"
            f"{json.dumps(world_json, ensure_ascii=True, indent=2)}\n\n"
            "inventory_json:\n"
            f"{inv_block}\n\n"
            "history_pseudo_function (if any):\n"
            f"{loop_history}\n\n"
            "critic_feedback_history (if any):\n"
            f"{feedback_history}\n\n"
        }
    ]
    plan = call_ollama(
        messages,
        model=PLANNER_MODEL,                 # ← VLM 사용
        temperature=TEMPERATURE_PLANNER,
    )
    pseudo_function = extract_pseudo_function(plan)
    if not pseudo_function:
        raise ValueError("planner가 pseudo_function을 생성하지 않았습니다.")
    return plan, pseudo_function

def critic(                                                                         # pseudo_function이 합리적인지 검사
        Critic_Prompt: str,
        user_instruction: str,
        world_json: dict,
        pseudo_function: list[str],
        history_pseudo_function: list[str] | None = None,
        inventory_json: dict | None = None,
):
    if history_pseudo_function:
        loop_history = "\n\n".join(
            f"Loop {i}\n```python\n{code.strip()}\n```"
            for i, code in enumerate(history_pseudo_function, 1)
        )
    else:
        loop_history = "[None yet]"
    
    inv_block = json.dumps(inventory_json or {"holding":[]}, ensure_ascii=True, indent=2)

    messages = [
        {"role": "system", "content": Critic_Prompt},
        {"role": "user",
         "content":
            "User instruction:\n"
            f"{user_instruction}\n\n"
            "world_json (ground-truth):\n"
            f"{json.dumps(world_json, ensure_ascii=True, indent=2)}\n\n"
            "inventory_json:\n"
            f"{inv_block}\n\n"
            "Generated pseudo_function:\n"
            "```python\n" + "\n".join(pseudo_function) + "\n```\n\n"
            "history_pseudo_function (if any):\n"
            f"{loop_history}\n\n"
        }
    ]
    critique_raw = call_ollama(
        messages,
        model=CRITIC_MODEL,
        temperature=TEMPERATURE_CRITIC,
    )
    critique = clean_critic_feedback(critique_raw)
    return critique

def _strip_xml_block(text: str, tag: str) -> str:
    """<tag> ... </tag> 블록을 통째로 제거 (대소문자 무시, 멀티라인)."""
    pattern = rf"<\s*{tag}\s*>.*?<\s*/\s*{tag}\s*>"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

def clean_critic_feedback(raw: str) -> str:
    """
    Critic가 반환한 원문에서 숨은 사고/추론 블록 제거.
    - <think>...</think> 제거
    """
    s = raw
    # 불필요한 내부 사고 블록 제거
    for tag in ("think", "reflection", "reasoning"):
        s = _strip_xml_block(s, tag)

    s = s.strip()
    return s

# critiqe에 revise 유무 체크 시 사용
_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(ACCEPTED|REVISE REQUIRED)", re.I)

def needs_revision(critique: str) -> bool:                                          # critic의 결과가 accepted인지 revise인지 판별
    text = (critique or "").strip()
    m = _VERDICT_RE.search(text)
    if m:
        return m.group(1).upper().startswith("REVISE")
    return "REVISE" in text.upper()

def locate_search(pseudo_function: list[str]) -> Tuple[bool, Optional[int]]:        # pseudo_function에 search() 함수가 있는지 판별
    for i, line in enumerate(pseudo_function):
        if line.strip().startswith('search('):
            return True, i
    return False, None

def main(args=None):
    rclpy.init(args=args)
    node = VlmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
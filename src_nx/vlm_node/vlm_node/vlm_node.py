#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
planner_node.py — ROS 2 Humble node version of your original script.

Key points:
- Subscribes to "slam/json" (std_msgs/String) where each message is a JSON array or {"objects":[...]}.
- Publishes:
  * planner/plan_raw        (std_msgs/String)  — full LLM planner output text
  * planner/pseudo_lines    (std_msgs/String)  — newline-joined pseudo_function lines
  * planner/critique        (std_msgs/String)  — critic feedback
  * planner/world_json      (std_msgs/String)  — current world model snapshot
  * planner/inventory_json  (std_msgs/String)  — current inventory snapshot
  * planner/status          (std_msgs/String)  — high-level status updates
  * planner/exec_log        (std_msgs/String)  — execution/guard logs

- Parameters (declare via YAML/CLI):
  * user_instruction (string) — mission text
  * max_loops (int) — safety cap for planning iterations (default 20)
  * exec_delay_sec (double) — per-action simulated delay (default 5.0)
  * planner_model, critic_model, temperature_planner, temperature_critic
  * scenario_path (string) — optional path to JSON file for self-replay testing
  * scenario_tick_sec (double) — replay cadence (default 1.0)

Behavior:
- Each incoming SLAM JSON triggers one planning iteration, preserving history/inventory across cycles.
- If the generated plan contains search(...), the node executes lines up to (but not including) anything after search()
  and waits for the next SLAM update to continue (same as your original loop).
- If there’s no search(...) in the plan, it executes all steps and marks mission complete.
"""
import numpy as np
import cv2
from cv_bridge import CvBridge
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List, Optional
import json, re, ollama, time, sys, select
import os, sys, select
import base64
from typing import List, Optional
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
#----------------------------------------------------------추가한 부분
def occupancy_to_png_bytes(grid) -> bytes:
    """nav_msgs/OccupancyGrid → PNG 바이트"""
    w = grid.info.width
    h = grid.info.height
    data = np.asarray(grid.data, dtype=np.int16).reshape((h, w))
    # -1(unknown)=128, 0(자유)=255, 100(점유)=0 로 매핑
    img = np.full((h, w), 128, dtype=np.uint8)
    img[data == 0] = 255
    img[data == 100] = 0
    # 회전/flip은 필요시 추가 (RViz 와 축이 다를 때)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패(OccupancyGrid)")
    return buf.tobytes()

def cvimage_to_png_bytes(cvimg) -> bytes:
    ok, buf = cv2.imencode(".png", cvimg)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패(Image)")
    return buf.tobytes()
#-----------------------------------------------------------추가한 부분
# ─────────────────────────────────────────────────────────────
# 데이터 구조: 월드 모델(JSON)
# ─────────────────────────────────────────────────────────────
@dataclass
class ObjNode:
    id: str                         # SLAM에서 넘어온 고유 ID("cup#3")
    klass: str                      # 클래스명("cup")
    map_xy: Tuple[float, float]     # 맵 좌표계 절대좌표(x, y)

# ─────────────────────────────────────────────────────────────
# WorldModel helper: 다양한 JSON 포맷을 단일 objects 리스트로 정규화
# ─────────────────────────────────────────────────────────────
from typing import Any, Iterable

def _normalize_objects(payload: Any) -> list[dict]:
    """
    허용 포맷:
      - {"objects":[{id,class,map_xy}, ...]}
      - [{id,class,map_xy}, ...]
      - [{"objects":[{...}]}, {"objects":[{...}]}]
      - 혼합: [{"objects":[...]}, {id,class,map_xy}, ...]
    반환:
      - id/class/map_xy 가 모두 있는 dict 들만 담은 리스트
    """
    # 0) 문자열이면 파싱
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return []

    # 1) candidates: 오브젝트 dict들을 “후보”로 추출
    candidates: list[dict] = []

    def _yield_objs(x: Any) -> Iterable[dict]:
        # dict인 경우
        if isinstance(x, dict):
            if "objects" in x and isinstance(x["objects"], list):
                for o in x["objects"]:
                    if isinstance(o, dict):
                        yield o
            else:
                # 단일 오브젝트로 간주
                yield x
        # list인 경우: 각 원소를 재귀적으로 검사
        elif isinstance(x, list):
            for it in x:
                yield from _yield_objs(it)

    for obj in _yield_objs(payload):
        candidates.append(obj)

    # 2) 강력 필터: id/class/map_xy 존재 + map_xy 길이 ≥2
    out: list[dict] = []
    for n in candidates:
        try:
            if not isinstance(n, dict):
                continue
            if "id" not in n or "class" not in n or "map_xy" not in n:
                continue
            mxy = n["map_xy"]
            if not isinstance(mxy, (list, tuple)) or len(mxy) < 2:
                continue
            # 숫자 변환 가능성 확인
            _mx = float(mxy[0]); _my = float(mxy[1])
            # 통과
            out.append(n)
        except Exception:
            # 형식 이상은 조용히 드랍
            continue
    return out

@dataclass
class WorldModel:
    objects: Dict[str, ObjNode] = field(default_factory=dict)

    def apply_slam_json(self, slam_json: str | dict | list) -> int:
        """
        들어오는 모든 변형 포맷을 허용하고,
        유효(id/class/map_xy= [x,y])한 항목만 self.objects로 반영.
        """
        try:
            nodes = _normalize_objects(slam_json)
        except Exception:
            nodes = []

        self.objects.clear()
        updated = 0
        for n in nodes:
            # 여기선 이미 id/class/map_xy가 검증된 상태지만, 마지막까지 방어
            try:
                oid = str(n["id"])
                klass = str(n["class"])
                mx = float(n["map_xy"][0])
                my = float(n["map_xy"][1])
                self.objects[oid] = ObjNode(id=oid, klass=klass, map_xy=(mx, my))
                updated += 1
            except Exception:
                # 개별 항목 문제는 건너뜀(로그 필요 시 여기에 출력)
                continue
        return updated

    def to_json(self, indent: int | None = 2, ensure_ascii: bool = True) -> str:
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
- search(object): When there is no relevant object for the mission in world_json, it explores unexplored areas to find such an object. **You first need to search for the object first if the object you need to deal with is not in the world_json.**
- pick(object): Pick up the specified object.
- place(object): Put down the held object at the current location. Only write a single object you want to place. *You first need to go to the place where you want to place the object.* Example: If you want to place an apple on the table, call move_to(table) then place(apple).
- return_to_base(): Return to the robot's base location.

HARD RULES:
- List function calls only. Do not use control flow (if/else, for/while), variable definitions, imports, or any other Python syntax.
- ***Output nothing after search() in the same plan.***
- If an object required to complete the task is not present in the current world_json, call search().
- Use history_pseudo_function to avoid repeating already executed steps unless repetition is necessary to make progress.
- If an object appears in inventory_json.holding, treat it as currently held even if it is absent from world_json; do NOT call pick() for it. Prefer place() when appropriate.

OUTPUT FORMAT (STRICT):
1) Reasoning (plain text, 1-5 short bullet lines explaining the plan)
2) Pseudocode (exactly one fenced python block containing only pseudo_function, one per line)
"""

Critic_Prompt = """
You are a *generous* robot plan critic.
Your task is to evaluate the pseudo_function plans proposed by the planner.
If the score is insufficient, analyze the surrounding context and recommend the necessary steps and processes.

ALLOWED FUNCTIONS:
- move_to(object): Move toward the specified object or location.
- search(object): Use when an object required to complete the mission is not visible. **As an exception, search(object) may take an object that is not currently visible in the world_json as its argument.**
- pick(object): Pick up the specified object.
- place(object): Put down the held object at the current location.
- return_to_base(): Return to the robot's base location.

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
   - Placing search(object) last and referencing an unseen (but required) object is logically correct and should not reduce the score.
2. Context awareness — Does the plan correctly respond to present or known elements in the world_json?
   - When a key object is missing from the current JSON but recorded as acquired in history, infer that it remains in the robot’s possession.
   - Use the provided history_pseudo_function to infer such states.
   - Assuming absence for an object in possession is a context error.
   - Exception: for search(), it is allowed to reference objects that are not visible in the current world_json. **Do not deduct points for this; it is required behavior.**
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

Be generous. If the total score (Logical correctness + Context awareness + Completeness) exceeds **7**, set VERDICT: ACCEPTED; otherwise set VERDICT: REVISE REQUIRED.
"""

PLANNER_MODEL = "qwen2.5vl:7b"   # VLM (planner 용)
CRITIC_MODEL  = "qwen3:1.7b"     # LLM (critic 용)

TEMPERATURE_PLANNER = 0.7
TEMPERATURE_CRITIC  = 0.7

def call_ollama(
    messages: List[dict],
    model: str,
    temperature: float = 0.2,
    images_b64: Optional[List[str]] = None,
):
    # images_b64가 주어지면 마지막 user 메시지에 첨부
    if images_b64:
        # 뒤에서부터 첫 user 메시지 찾기
        for m in reversed(messages):
            if m.get("role") == "user":
                if isinstance(m.get("content"), str) and m.get("content", "").strip() == "":
                    # content가 비어있으면 프롬프트 누락 방지용 더미 텍스트
                    m["content"] = "[image attached]"
                # Ollama /api/chat 은 message에 images:[base64, ...] 허용
                m["images"] = list(images_b64)
                break
        else:
            # user 메시지가 없다면 새로 하나 추가
            messages.append({
                "role": "user",
                "content": "[image attached]",
                "images": list(images_b64),
            })

    resp = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return resp["message"]["content"]

def extract_pseudo_function(text: str) -> list[str] | None:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()
    return None

def read_image_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def planner(
    Actor_Prompt: str,
    user_instruction: str,
    world_json: dict,
    history_pseudo_function: list[str] | None = None,
    critic_feedbacks: str | None = None,
    inventory_json: dict | None = None,
    images_b64: Optional[List[str]] = None,   # [추가] 이미지 인자
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
        model=PLANNER_MODEL,
        temperature=TEMPERATURE_PLANNER,
        images_b64=images_b64,      # [추가] 이미지 전달
    )
    pseudo_function = extract_pseudo_function(plan)
    if not pseudo_function:
        raise ValueError("planner가 pseudo_function을 생성하지 않았습니다.")
    return plan, pseudo_function


def critic(
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
    pattern = rf"<\s*{tag}\s*>.*?<\s*/\s*{tag}\s*>"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

def clean_critic_feedback(raw: str) -> str:
    s = raw
    for tag in ("think", "reflection", "reasoning"):
        s = _strip_xml_block(s, tag)
    return s.strip()

_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(ACCEPTED|REVISE REQUIRED)", re.I)

def needs_revision(critique: str) -> bool:
    text = (critique or "").strip()
    m = _VERDICT_RE.search(text)
    if m:
        return m.group(1).upper().startswith("REVISE")
    return "REVISE" in text.upper()

def locate_search(pseudo_function: list[str]) -> Tuple[bool, Optional[int]]:
    pattern = re.compile(
        r'^\s*search\s*\(\s*(?:(?:"[^"]*"|\'[^\']*\'|\w+)\s*)?\)\s*;?\s*(#.*)?$',
        re.IGNORECASE
    )
    for idx, line in enumerate(pseudo_function, start=1):
        if pattern.match(line):
            return True, idx
    return False, None

def _parse_call(line: str) -> Tuple[str, Optional[str]]:
    m = re.match(r'^\s*(\w+)\s*\(\s*(.*?)\s*\)\s*;?\s*(?:#.*)?$', line, re.IGNORECASE)
    if not m:
        return "", None

    name = m.group(1)
    raw_arg = (m.group(2) or "").strip()

    if raw_arg == "":
        return name, None

    if len(raw_arg) >= 2 and raw_arg[0] == raw_arg[-1] and raw_arg[0] in ("'", '"'):
        return name, raw_arg[1:-1]

    return name, raw_arg

def _execute_pseudocode_lines(lines: list[str], delay_sec: float = 1.0, logger=lambda s: None):
    for ln in lines:
        name, arg = _parse_call(ln)
        if not name:
            logger(f"[EXEC] 무시: {ln!r}")
            continue

        disp_arg = f'("{arg}")' if arg is not None else "()"
        logger(f"[EXEC] {name}{disp_arg} 호출")
        time.sleep(max(0.0, float(delay_sec)))
        logger(f"[EXEC] {name}{disp_arg} 수행 완료")

def update_inventory_from_lines(lines: list[str], world: WorldModel, inv: InventoryModel, logger=lambda s: None) -> None:
    for ln in lines:
        name, arg = _parse_call(ln)
        if not name:
            continue
        name_l = name.lower()
        if name_l == "pick" and arg:
            ok = inv.add_from_world(world, arg)
            logger(f"[INV] pick('{arg}') → {'added' if ok else 'skip (ambiguous/not found)'}")
        elif name_l == "place" and arg:
            ok = inv.remove_by_ref(arg)
            logger(f"[INV] place('{arg}') → {'removed' if ok else 'skip (not in inventory)'}")

def _present_in_world(world: WorldModel, ref: str) -> bool:
    return (ref in world.objects) or any(n.klass == ref for n in world.objects.values())

def _present_in_inventory(inv: InventoryModel, ref: str) -> bool:
    return (ref in inv.items) or any(n.klass == ref for n in inv.items.values())

def validate_against_state(line: str, world: WorldModel, inv: InventoryModel):
    name, arg = _parse_call(line)
    if not name:
        return (True, "", "")

    nm = name.lower()
    if nm == "move_to":
        if arg is None or not _present_in_world(world, arg):
            return (False,
                    f"move_to target '{arg}' is not present in the current world_json. Consider calling search({arg}) first. "
                    f"Visible objects: {[f'{oid}:{node.klass}' for oid,node in world.objects.items()]}",
                    line)
    elif nm == "pick":
        if arg is None or not _present_in_world(world, arg):
            return (False,
                    f"pick target '{arg}' is not present in the current world_json. Consider calling search({arg}) first. "
                    f"Visible objects: {[f'{oid}:{node.klass}' for oid,node in world.objects.items()]}",
                    line)
    elif nm == "place":
        if arg is None or not _present_in_inventory(inv, arg):
            return (False,
                    f"place target '{arg}' is not present in inventory_json.holding. "
                    f"Currently holding: {[f'{oid}:{node.klass}' for oid,node in inv.items.items()]}",
                    line)
    elif nm == "search":
        return (True, "", "")
    return (True, "", "")

def _guard_pseudo_function(pseudo_lines: list[str], world: WorldModel, inv: InventoryModel) -> tuple[bool, str]:
    inv_sim = InventoryModel(items=dict(inv.items))  # 실제 인벤토리 보호

    for ln in pseudo_lines:
        ok, msg, bad_ln = validate_against_state(ln, world, inv_sim)
        if not ok:
            return False, f"[GUARD] invalid: {bad_ln}\n{msg}"

        name, _ = _parse_call(ln)
        if name and name.lower() == "search":
            return True, ""

        update_inventory_from_lines([ln], world, inv_sim)

    return True, ""

# ─────────────────────────────────────────────────────────────
# ROS 2 Node
# ─────────────────────────────────────────────────────────────
class PlannerNode(Node):
    def __init__(self):
        global PLANNER_MODEL, CRITIC_MODEL, TEMPERATURE_PLANNER, TEMPERATURE_CRITIC

        super().__init__("llm_planner_node")

        # Parameters
        self.declare_parameter("user_instruction", "Pick up the laundry and put it into the washing machine.")
        self.declare_parameter("max_loops", 20)
        self.declare_parameter("exec_delay_sec", 5.0)
        self.declare_parameter("planner_model", PLANNER_MODEL)
        self.declare_parameter("critic_model", CRITIC_MODEL)
        self.declare_parameter("temperature_planner", TEMPERATURE_PLANNER)
        self.declare_parameter("temperature_critic", TEMPERATURE_CRITIC)
        self.declare_parameter("scenario_path", "/home/nx/nx_ws/src/vlm_node/vlm_node/scenario_1.json")
        self.declare_parameter("scenario_tick_sec", 1.0)
        # 기존 declare_parameter들 옆에 추가
        self.declare_parameter("use_merged_map_sub", True)     # merged_map 구독 On/Off
        self.declare_parameter("merged_map_topic", "merged_map")
        self.declare_parameter("merged_map_is_image", False)   # True면 sensor_msgs/Image, False면 OccupancyGrid

        # 멤버 초기화
        self.bridge = CvBridge()
        self.images_b64 = None            # 기존 파일 로딩과 공유
        self._last_image_stamp = None


        # Pull params into module-level globals the original functions use
        PLANNER_MODEL = self.get_parameter("planner_model").get_parameter_value().string_value
        CRITIC_MODEL  = self.get_parameter("critic_model").get_parameter_value().string_value
        TEMPERATURE_PLANNER = float(self.get_parameter("temperature_planner").value)
        TEMPERATURE_CRITIC  = float(self.get_parameter("temperature_critic").value)

        self.user_instruction = self.get_parameter("user_instruction").get_parameter_value().string_value
        self.get_logger().info(self.user_instruction)
        self.max_loops = int(self.get_parameter("max_loops").value)
        self.exec_delay_sec = float(self.get_parameter("exec_delay_sec").value)

        # Publishers
        self.pub_plan_raw       = self.create_publisher(String, "planner/plan_raw", 10)
        self.pub_pseudo_lines   = self.create_publisher(String, "planner/pseudo_lines", 10)
        self.pub_critique       = self.create_publisher(String, "planner/critique", 10)
        self.pub_world_json     = self.create_publisher(String, "planner/world_json", 10)
        self.pub_inventory_json = self.create_publisher(String, "planner/inventory_json", 10)
        self.pub_status         = self.create_publisher(String, "planner/status", 10)
        self.pub_exec_log       = self.create_publisher(String, "planner/exec_log", 10)
        # NEW: publish search target / end-signal
        self.pub_search_signal  = self.create_publisher(String, "planner/search_signal", 10)

        # Subscriber
        # self.sub_slam = self.create_subscription(String, "slam/json", self._on_slam_msg, 10)
        self.sub_slam = self.create_subscription(String, "detected_objects_json", self._on_slam_msg, 10)

        # State
        self.world = WorldModel()
        self.inventory = InventoryModel()
        self.history_blocks: list[str] = []
        self.loop_idx = 0
        self.mission_complete = False
        # single-threaded busy flag (no threading)
        self._busy: bool = False
        # search gating + stdin poller (no threads)
        self.awaiting_search: Optional[str] = None
        self._stdin_timer = self.create_timer(0.05, self._poll_stdin)  # 20 Hz


#-----------------------------------------------------------------------------------------------------추가한 부분
        # (파일 경로 base64 로딩 로직이 이미 있다면 그대로 두고, 아래 구독이 있으면 최신 프레임으로 갱신)

        # 구독 설정
        if bool(self.get_parameter("use_merged_map_sub").value):
            topic = self.get_parameter("merged_map_topic").get_parameter_value().string_value
            is_img = bool(self.get_parameter("merged_map_is_image").value)
            if is_img:
                from sensor_msgs.msg import Image as RosImage
                self.sub_map = self.create_subscription(RosImage, topic, self._cb_merged_map_image, 10)
                self._emit_status(f"[MERGED_MAP] Subscribed (Image): {topic}")
            else:
                from nav_msgs.msg import OccupancyGrid
                self.sub_map = self.create_subscription(OccupancyGrid, topic, self._cb_merged_map_grid, 10)
                self._emit_status(f"[MERGED_MAP] Subscribed (OccupancyGrid): {topic}")
        # 빈 문자열이면 이미지 미사용
#-----------------------------------------------------------------------------------------------------추가한 부분
        # Optional scenario replay (for testing without a live SLAM topic)
        scenario_path = self.get_parameter("scenario_path").get_parameter_value().string_value
        self._scenario_data: Optional[List[Any]] = None
        self._scenario_i = 0
        if scenario_path:
            try:
                with open(scenario_path, "r", encoding="utf-8") as f:
                    self._scenario_data = json.load(f)
                tick = float(self.get_parameter("scenario_tick_sec").value)
                self.create_timer(tick, self._scenario_tick_cb)
                self._emit_status(f"[SCENARIO] Loaded {len(self._scenario_data)} frames from: {scenario_path}")
            except Exception as e:
                self._emit_status(f"[SCENARIO] Failed to load: {scenario_path} ({e})")
        ## -----------------------------------------------------------------------------------추가한 부분
        self.images_b64: Optional[List[str]] = None
        img_path = self.get_parameter("input_image_path").get_parameter_value().string_value
        if img_path:
            if os.path.exists(img_path):
                try:
                    img_b64 = read_image_as_base64(img_path)
                    self.images_b64 = [img_b64]
                    self._emit_status(f"[IMAGE] Loaded image: {img_path}")
                except Exception as e:
                    self._emit_status(f"[IMAGE] Failed to load image: {img_path} ({e})")
            else:
                self._emit_status(f"[IMAGE] Not found: {img_path}")
        ## -----------------------------------------------------------------------------------추가한 부분
        self._emit_status("Planner node ready.")

    def _cb_merged_map_image(self, msg):
        try:
            cvimg = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            png_bytes = cvimage_to_png_bytes(cvimg)
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            self.images_b64 = [b64]         # planner()가 매 루프에서 사용
            self._last_image_stamp = msg.header.stamp
        except Exception as e:
            self._emit_status(f"[MERGED_MAP][Image] 변환 실패: {e}")

    def _cb_merged_map_grid(self, msg):
        try:
            png_bytes = occupancy_to_png_bytes(msg)
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            self.images_b64 = [b64]
            self._last_image_stamp = msg.info.map_load_time  # 또는 rclpy.time.Time()
        except Exception as e:
            self._emit_status(f"[MERGED_MAP][OccupancyGrid] 변환 실패: {e}")

    # ─────────────────────────────
    # Helpers to publish logs/status
    # ─────────────────────────────
    def _emit_exec(self, msg: str):
        self.get_logger().info(msg)
        self.pub_exec_log.publish(String(data=msg))

    def _emit_status(self, msg: str):
        self.get_logger().info(msg)
        self.pub_status.publish(String(data=msg))

    # NEW: non-blocking stdin poller (single-threaded)
    def _poll_stdin(self):
        # Only react while we're intentionally paused after search(...)
        if self.awaiting_search is None:
            return
        # If stdin is not available (e.g., ros2 launch), publish X once and unpause.
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception as e:
            self.pub_search_signal.publish(String(data="X"))
            self._emit_status(f"[SEARCH] stdin unavailable ({e}); published 'X' and resuming.")
            self.awaiting_search = None
            return
        if not r:
            return
        line = sys.stdin.readline()
        if line == "":  # EOF
            self.pub_search_signal.publish(String(data="X"))
            self._emit_status("[SEARCH] stdin EOF; published 'X' and resuming.")
            self.awaiting_search = None
            return
        s = line.strip().lower()
        if s == "end":
            self.pub_search_signal.publish(String(data="X"))
            self._emit_status("[SEARCH] Received 'end' → published 'X' and resuming.")
            self.awaiting_search = None
        else:
            # Any other input publishes the search target once (object name)
            target = str(self.awaiting_search or "")
            self.pub_search_signal.publish(String(data=target))
            self._emit_status(f"[SEARCH] Published '{target}' on planner/search_signal. "
                              f"Type 'end' to finish search and resume.")

    # Scenario timer — feeds frames into the same processing path
    def _scenario_tick_cb(self):
        if self.mission_complete or not self._scenario_data:
            return
        if self.loop_idx >= self.max_loops:
            self._emit_status("Reached max_loops — stopping scenario replay.")
            return
        if self._scenario_i >= len(self._scenario_data):
            self._emit_status("Scenario frames exhausted.")
            return
        frame = self._scenario_data[self._scenario_i]
        self._scenario_i += 1
        self._on_slam(frame)  # direct (dict) path

    # ROS sub callback
    def _on_slam_msg(self, msg: String):
        try:
            slam_obj = json.loads(msg.data)
        except Exception:
            slam_obj = msg.data
        self._on_slam(slam_obj)

    # Common entry (dict or str)
    def _on_slam(self, slam_json: Any):
        self.get_logger().info(self.user_instruction)
        if self.mission_complete:
            return
        if self.loop_idx >= self.max_loops:
            self._emit_status("최대 루프 수에 도달했습니다. 미션 실패.")
            return
        if self._busy:
            self._emit_exec("[RUN] Busy; skipping incoming SLAM frame.")
            return
        # If we're waiting for terminal input after search(...), do not plan yet.
        if self.awaiting_search is not None:
            # (Optional) still refresh world snapshot so UI can see changes
            self.world.objects.clear()
            self.world.apply_slam_json(slam_json)
            self.pub_world_json.publish(String(data=self.world.to_json(indent=None, ensure_ascii=False)))
            self._emit_status("[RUN] Waiting for terminal input to proceed after search(...). Planning paused.")
            return

        try:
            self._busy = True
            self.loop_idx += 1
            self._emit_status(f"\nLoop {self.loop_idx} 시작")

            # SLAM 반영 및 world_json 구성 (visible-only each loop)
            self.world.objects.clear()
            updated = self.world.apply_slam_json(slam_json)
            self._emit_exec(f"[RUN] SLAM 반영: {updated}개 객체 업데이트")

            world_json_dict = json.loads(self.world.to_json(indent=None))
            self.pub_world_json.publish(String(data=json.dumps(world_json_dict, ensure_ascii=False)))

            # 1) Initial plan
            try:
                plan, pseudo_lines = planner(
                    Actor_Prompt=Actor_Prompt,
                    user_instruction=self.user_instruction,
                    world_json=world_json_dict,
                    history_pseudo_function=self.history_blocks,
                    critic_feedbacks=None,
                    inventory_json=self.inventory.to_dict(),
                    images_b64=self.images_b64,
                )
            except Exception as e:
                self._emit_status(f"[ERROR] planner failed: {e}")
                return

            self.pub_plan_raw.publish(String(data=plan))
            self.pub_pseudo_lines.publish(String(data="\n".join(pseudo_lines)))
            self._emit_exec("\n===== PLAN (initial) =====\n" + plan + "\n=========================\n")

            # Guard & replan if needed
            MAX_GUARD_REPLANS = 5
            guard_tries = 0
            ok_guard, guard_msg = _guard_pseudo_function(pseudo_lines, self.world, self.inventory)
            while (not ok_guard) and (guard_tries < MAX_GUARD_REPLANS):
                self._emit_exec(f"[RUN] Guard 실패 → 재계획 시도 {guard_tries+1}\n{guard_msg}\n")
                plan, pseudo_lines = planner(
                    Actor_Prompt=Actor_Prompt,
                    user_instruction=self.user_instruction,
                    world_json=world_json_dict,
                    history_pseudo_function=self.history_blocks,
                    critic_feedbacks=guard_msg,
                    inventory_json=self.inventory.to_dict(),
                    images_b64=self.images_b64,
                )
                self.pub_plan_raw.publish(String(data=plan))
                self.pub_pseudo_lines.publish(String(data="\n".join(pseudo_lines)))
                self._emit_exec(f"\n===== PLAN (guard replan {guard_tries+1}) =====\n{plan}\n==============================================\n")
                ok_guard, guard_msg = _guard_pseudo_function(pseudo_lines, self.world, self.inventory)
                guard_tries += 1

            # Critic
            MAX_CRITIC_REPLANS = 5
            replan_tries = 0
            critique = critic(
                Critic_Prompt=Critic_Prompt,
                user_instruction=self.user_instruction,
                world_json=world_json_dict,
                pseudo_function=pseudo_lines,
                history_pseudo_function=self.history_blocks,
                inventory_json=self.inventory.to_dict(),
            )
            self.pub_critique.publish(String(data=critique))
            self._emit_exec(f"\n===== CRITIQUE =====\n{critique}\n==============================================\n")

            while needs_revision(critique) and (replan_tries < MAX_CRITIC_REPLANS):
                self._emit_exec(f"[RUN] Critic: REVISE REQUIRED → 재계획 시도 {replan_tries+1}")
                plan, pseudo_lines = planner(
                    Actor_Prompt=Actor_Prompt,
                    user_instruction=self.user_instruction,
                    world_json=world_json_dict,
                    history_pseudo_function=self.history_blocks,
                    critic_feedbacks=critique,
                    inventory_json=self.inventory.to_dict(),
                    images_b64=self.images_b64,
                )
                self.pub_plan_raw.publish(String(data=plan))
                self.pub_pseudo_lines.publish(String(data="\n".join(pseudo_lines)))
                self._emit_exec(f"\n===== PLAN (replan try {replan_tries+1}) =====\n{plan}\n=============================================\n")

                # Guard again within critic loop
                guard_tries_crit = 0
                ok_guard, guard_msg = _guard_pseudo_function(pseudo_lines, self.world, self.inventory)
                while (not ok_guard) and (guard_tries_crit < MAX_GUARD_REPLANS):
                    self._emit_exec(f"[RUN] Guard(critic 단계) 실패 → 재계획 시도 {guard_tries_crit+1}\n{guard_msg}\n")
                    plan, pseudo_lines = planner(
                        Actor_Prompt=Actor_Prompt,
                        user_instruction=self.user_instruction,
                        world_json=world_json_dict,
                        history_pseudo_function=self.history_blocks,
                        critic_feedbacks=guard_msg,
                        inventory_json=self.inventory.to_dict(),
                        images_b64=self.images_b64,
                    )
                    self.pub_plan_raw.publish(String(data=plan))
                    self.pub_pseudo_lines.publish(String(data="\n".join(pseudo_lines)))
                    self._emit_exec(f"\n===== PLAN (guard@critic replan {guard_tries_crit+1}) =====\n{plan}\n===========================================================\n")
                    ok_guard, guard_msg = _guard_pseudo_function(pseudo_lines, self.world, self.inventory)
                    guard_tries_crit += 1

                if not ok_guard:
                    self._emit_exec("[RUN] Guard(critic 단계) 재계획 한도 도달 → 현재 계획으로 Critic 평가 진행")

                critique = critic(
                    Critic_Prompt=Critic_Prompt,
                    user_instruction=self.user_instruction,
                    world_json=world_json_dict,
                    pseudo_function=pseudo_lines,
                    history_pseudo_function=self.history_blocks,
                    inventory_json=self.inventory.to_dict(),
                )
                self.pub_critique.publish(String(data=critique))
                self._emit_exec(f"\n===== CRITIQUE =====\n{critique}\n==============================================\n")

                replan_tries += 1
                if needs_revision(critique) and (replan_tries >= MAX_CRITIC_REPLANS):
                    self._emit_exec("[RUN] 재계획 한도 도달 → 현재 계획으로 진행")
                    break

            # 4) search() 유무 확인
            has_search, search_line = locate_search(pseudo_lines)

            if has_search:
                executed = pseudo_lines[:search_line]
                for ln in executed:
                    _execute_pseudocode_lines([ln], delay_sec=self.exec_delay_sec, logger=self._emit_exec)
                    update_inventory_from_lines([ln], self.world, self.inventory, logger=self._emit_exec)

                # NEW: extract search target and start stdin-pause mode
                try:
                    _, arg = _parse_call(pseudo_lines[search_line-1])
                except Exception:
                    arg = None
                self.awaiting_search = arg
                pretty = arg or ""
                self._emit_status(f"[SEARCH] Planned search('{pretty}'). "
                                  f"Type anything to publish '{pretty}', or 'end' to publish 'X' and resume.")

                self.history_blocks.append("\n".join(pseudo_lines))
                self.pub_inventory_json.publish(String(data=self.inventory.to_json(indent=2, ensure_ascii=False)))
                self._emit_status("[RUN] Paused at search(). Waiting for terminal input…")
                return  # stay paused until _poll_stdin() clears awaiting_search

            # 5) search()가 없으면 전체 실행 → 미션 완료
            for ln in pseudo_lines:
                _execute_pseudocode_lines([ln], delay_sec=self.exec_delay_sec, logger=self._emit_exec)
                update_inventory_from_lines([ln], self.world, self.inventory, logger=self._emit_exec)

            self.history_blocks.append("\n".join(pseudo_lines))
            self.pub_inventory_json.publish(String(data=self.inventory.to_json(indent=2, ensure_ascii=False)))
            self._emit_status("[RUN] Mission complete.")
            self.mission_complete = True

        finally:
            self._busy = False

def main():
    rclpy.init()
    node = PlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List, Optional
import json, re, ollama, time, math
# 맨 위 import 구역에
import base64
import numpy as np
import cv2

from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from rclpy.qos import QoSPresetProfiles, QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# ─────────────────────────────────────────────────────────────
# ROS2 (Humble) minimal additions
# ─────────────────────────────────────────────────────────────
import threading, queue
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String as StringMsg
from std_msgs.msg import Bool as BoolMsg
from geometry_msgs.msg import PoseStamped

# ─────────────────────────────────────────────────────────────
# 데이터 구조: 월드 모델(JSON)
# ─────────────────────────────────────────────────────────────
@dataclass
class ObjNode:
    id: str                         # SLAM에서 넘어온 고유 ID("cup#3")
    klass: str                      # 클래스명("cup")
    map_xy: Tuple[float, float]     # 맵 좌표계 절대좌표(x, y)

from typing import Iterable

def _iter_objects(payload: Any) -> Iterable[dict]:
    """
    들어온 payload에서 실제 object 딕셔너리들만 yield.
    지원 케이스:
      1) dict: {"objects":[{...}, {...}]}
      2) list: [{"objects":[...]} , {"objects":[...]}]
      3) list: [{id, class, map_xy}, {...}]  # 바로 object들의 리스트
      4) str : 위의 어떤 형태든 JSON 문자열로 들어옴
    그 외(형식 불일치)는 빈 iterator 반환.
    """
    # 4) 문자열이면 먼저 JSON 파싱
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except Exception:
            return  # 잘못된 JSON 문자열이면 그냥 무시
        yield from _iter_objects(decoded)
        return

    # 1) dict
    if isinstance(payload, dict):
        # {"objects":[...]} 패턴
        if "objects" in payload and isinstance(payload["objects"], list):
            for o in payload["objects"]:
                # 오브젝트 구조 자체가 dict여야 함
                if isinstance(o, dict):
                    yield o
            return
        # 혹시 단일 object(dict) 자체가 온 경우(id/class/map_xy 3요소 모두 포함 시 처리)
        if all(k in payload for k in ("id", "class", "map_xy")):
            yield payload
        return

    # 2) 또는 3) list
    if isinstance(payload, list):
        for item in payload:
            # 리스트 안의 항목들에 대해 재귀적으로 objects만 추출
            yield from _iter_objects(item)
        return

@dataclass
class WorldModel:
    objects: Dict[str, ObjNode] = field(default_factory=dict)

    def apply_slam_json(self, slam_json: str | dict | list) -> int:  # [변경] list도 타입 힌트에 추가
        """
        어떤 형태로 들어와도 objects만 뽑아 넣는다.
        - {"objects":[...]} 또는 [{"objects":[...]} , ...] 또는 [{id,...}, ...] 모두 허용
        - 문자열로 오면 내부에서 json.loads 후 재귀 정규화
        """
        updated = 0
        for n in _iter_objects(slam_json):  # [변경] 정규화된 object들만 순회
            try:
                oid = str(n["id"])
                klass = str(n["class"])
                mx, my = n["map_xy"]
                self.objects[oid] = ObjNode(
                    id=oid,
                    klass=klass,
                    map_xy=(float(mx), float(my)),
                )
                updated += 1
            except Exception:
                # 개별 오브젝트에 필드가 빠져 있으면 그 항목만 건너뜀
                continue
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
# 프롬프트: 외부 텍스트 파일에서 읽기
# ─────────────────────────────────────────────────────────────
ACTOR_PROMPT_FILE = Path("/home/nx/nx_ws/src/vlm/vlm/prompts/actor_prompt.txt")
CRITIC_PROMPT_FILE = Path("/home/nx/nx_ws/src/vlm/vlm/prompts/critic_prompt_test.txt")

def _read_prompt(path: Path) -> str:
    # 파일이 없으면 즉시 알리는 편이 디버깅에 유리합니다.
    return path.read_text(encoding="utf-8")

Actor_Prompt = _read_prompt(ACTOR_PROMPT_FILE)
Critic_Prompt = _read_prompt(CRITIC_PROMPT_FILE)


# ─────────────────────────────────────────────────────────────
# LLM 백엔드
# ─────────────────────────────────────────────────────────────

PLANNER_MODEL = "qwen2.5vl:7b"   # VLM (planner 용)
CRITIC_MODEL  = "qwen2.5vl:7b"     # LLM (critic 용)

temp = 0.2
TEMPERATURE_PLANNER = temp
TEMPERATURE_CRITIC  = temp

def call_ollama(messages, model=str, temperature: float = 0.2):                    # ollama 모델을 띄움
    resp = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return resp["message"]["content"]

def extract_pseudo_function(text: str) -> list[str] | None:                         # LLM이 짜준 plan에서 함수 부분만 리스트로 저장
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()
    return None

def planner(                                                                        # LLM이 plan 생성
    Actor_Prompt: str,
    user_instruction: str,
    world_json: dict,
    history_pseudo_function: list[str] | None = None,
    critic_feedbacks: str | None = None,
    inventory_json: dict | None = None,
    image_b64: str | None = None,
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

    # [변경] user 메시지를 dict로 만들고, image_b64가 있으면 images 필드 추가
    user_msg = {
        "role": "user",
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
    }  # [추가]

    if image_b64:                               # [추가] base64 이미지 첨부 (Ollama VLM 규격)
        user_msg["images"] = [image_b64]

    messages = [
        {"role": "system", "content": Actor_Prompt},
        user_msg,                               # [변경] 위에서 구성한 user_msg 사용
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
        plan: list[str],
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
            f"{plan}\n\n"
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
    pattern = re.compile(
        r'^\s*search\s*\(\s*(?:(?:"[^"]*"|\'[^\']*\'|\w+)\s*(?:,\s*(?:"[^"]*"|\'[^\']*\'|\w+)\s*)*)?\)\s*;?\s*(#.*)?$',
        re.IGNORECASE
    )
    for idx, line in enumerate(pseudo_function, start=1):
        if pattern.match(line):
            return True, idx
    return False, None

def _parse_call(line: str) -> Tuple[str, Optional[str]]:                            # 실행시킬 함수 리스트를 파싱하는 함수
    m = re.match(r'^\s*(\w+)\s*\(\s*(.*?)\s*\)\s*;?\s*(?:#.*)?$', line, re.IGNORECASE)
    if not m:
        return "", None

    name = m.group(1)
    raw_arg = (m.group(2) or "").strip()

    if raw_arg == "":
        return name, None

    # 인자가 따옴표로 둘러싸여 있으면 제거
    if len(raw_arg) >= 2 and raw_arg[0] == raw_arg[-1] and raw_arg[0] in ("'", '"'):
        return name, raw_arg[1:-1]

    return name, raw_arg

def _resolve_map_xy(world: WorldModel, ref: str) -> Optional[Tuple[float, float]]:
    if ref in world.objects:
        return world.objects[ref].map_xy
    matches = [n for n in world.objects.values() if n.klass == ref]
    if not matches:
        return None
    matches.sort(key=lambda n: n.id)
    return matches[0].map_xy

# 터미널에서 'end' 대기
def _wait_for_terminal_end(prompt: str = "[INPUT] 다음 단계로 진행하려면 end 입력: "):
    while True:
        try:
            s = input(prompt).strip()
        except EOFError:
            s = "end"
        if s.lower() == "end":
            break

def _execute_pseudocode_lines(lines: list[str], world: WorldModel, ros_node: "DetectedObjectsNode") -> None:
    for ln in lines:
        name, arg = _parse_call(ln)
        if not name:
            print(f"[EXEC] 무시: {ln!r}")
            continue

        nm = name.lower()
        disp_arg = f'("{arg}")' if arg is not None else "()"
        print(f"[EXEC] {name}{disp_arg} 호출")

        if nm == "search":
            # 1) 키워드 pub → 2) 터미널 'end' 대기
            key = arg or ""
            ros_node.publish_keywords(key)
            print(f"[EXEC] search: '{key}' → '{ros_node.keywords_topic}' publish 완료")
            _wait_for_terminal_end()
            print(f"[EXEC] {name}{disp_arg} 수행 완료")
            continue

        if nm in ("pick", "place"):
            # 외부 실제 작업 후 사용자가 end 입력하면 진행
            _wait_for_terminal_end()
            print(f"[EXEC] {name}{disp_arg} 수행 완료")
            continue

        if nm == "move_to":
            if arg is None:
                print("[EXEC] move_to: 인자가 없음 → 스킵")
                continue
            xy = _resolve_map_xy(world, arg)
            if not xy:
                print(f"[EXEC] move_to: '{arg}' 좌표를 world에서 찾지 못함 → 스킵")
                continue
            x, y = xy
            ros_node.reset_control_complete()
            ros_node.publish_goal(x, y, yaw=0.0)
            print(f"[EXEC] move_to: goal ({x:.3f}, {y:.3f}) publish. control_compl=True 대기 중...")
            ros_node.wait_for_control_complete()
            print(f"[EXEC] {name}{disp_arg} 수행 완료")
            continue

        if nm == "return_to_base":
            ros_node.reset_control_complete()
            ros_node.publish_goal(0.0, 0.0, yaw=0.0)
            print(f"[EXEC] return_to_base: goal (0,0) publish. control_compl=True 대기 중...")
            ros_node.wait_for_control_complete()
            print(f"[EXEC] {name}{disp_arg} 수행 완료")
            continue

        # 기타 함수
        time.sleep(0.2)
        print(f"[EXEC] {name}{disp_arg} 수행 완료")

# move_to/search/return_to_base/speak 은 인벤토리 변화 없음
def update_inventory_from_lines(lines: list[str], world: WorldModel, inv: InventoryModel) -> None:
    for ln in lines:
        name, arg = _parse_call(ln)
        if not name:
            continue
        name_l = name.lower()
        if name_l == "pick" and arg:
            ok = inv.add_from_world(world, arg)
            print(f"[INV] pick('{arg}') → {'added' if ok else 'skip (ambiguous/not found)'}")
        elif name_l == "place" and arg:
            ok = inv.remove_by_ref(arg)
            print(f"[INV] place('{arg}') → {'removed' if ok else 'skip (not in inventory)'}")

def _present_in_world(world: WorldModel, ref: str) -> bool:
    # id 일치 또는 class 일치(하나라도 존재하면 이동 대상 존재로 간주)
    return (ref in world.objects) or any(n.klass == ref for n in world.objects.values())

def _present_in_inventory(inv: InventoryModel, ref: str) -> bool:
    # id 일치 또는 class 일치(들고 있는 것 중 하나라도 해당 클래스면 OK)
    return (ref in inv.items) or any(n.klass == ref for n in inv.items.values())

def validate_against_state(line: str, world: WorldModel, inv: InventoryModel):
    """
    단일 라인 상태 검증.
    - 성공: (True, "", "")
    - 실패: (False, 오류메시지, 문제라인문자열)
    """
    name, arg = _parse_call(line)
    if not name:
        return (True, "", "")

    nm = name.lower()
    if nm == "move_to":
        if arg is None or not _present_in_world(world, arg):
            return (False,
                    f"move_to target '{arg}' is not present in the current world_json. "
                    f"Visible objects: {[f'{oid}:{node.klass}' for oid,node in world.objects.items()]}",
                    line)
    elif nm == "pick":
        if arg is None or not _present_in_world(world, arg):
            return (False,
                    f"pick target '{arg}' is not present in the current world_json. "
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
    """
    한 줄씩 검증하고, 통과한 줄만 인벤토리 '시뮬레이션' 갱신.
    실패 시 (False, 메시지), 성공 시 (True, "").
    """
    inv_sim = InventoryModel(items=dict(inv.items))  # 실제 인벤토리 보호

    for ln in pseudo_lines:
        ok, msg, bad_ln = validate_against_state(ln, world, inv_sim)
        if not ok:
            return False, f"[GUARD] invalid: {bad_ln}\n{msg}"

        name, _ = _parse_call(ln)
        if name and name.lower() == "search":
            # search() 만나면 현재 루프는 여기까지 OK
            return True, ""

        # pick/place 반영
        update_inventory_from_lines([ln], world, inv_sim)

    return True, ""

def run(                                                                            # 전체 run 함수
    user_instruction: str,
    slam_stream,
    max_loops: int = 20,
    ros_node: Optional["DetectedObjectsNode"] = None,
):
    world = WorldModel()
    inventory = InventoryModel()
    history_blocks: list[str] = []

    for loop_idx, slam_json in enumerate(slam_stream, start=1):
        if loop_idx > max_loops:
            print("최대 루프 수에 도달했습니다. 미션 실패.")
            break

        print(f"\nLoop {loop_idx} 시작")

        # SLAM 반영 및 world_json 구성
        world.objects.clear()                                                       # 현재 보이는 object로만 pseudo_function 생성
        updated = world.apply_slam_json(slam_json)
        print(f"[RUN] SLAM 반영: {updated}개 객체 업데이트")
        world_json_dict = json.loads(world.to_json(indent=None))  # planner는 dict 기대

        # [추가] ROS 노드에서 최신 base64 이미지 가져오기
        _img_b64 = (ros_node.latest_image_b64
                    if (ros_node is not None and hasattr(ros_node, "latest_image_b64"))
                    else None)

        # 1차 계획
        plan, pseudo_lines = planner(
            Actor_Prompt=Actor_Prompt,
            user_instruction=user_instruction,
            world_json=world_json_dict,
            history_pseudo_function=history_blocks,
            critic_feedbacks=None,
            inventory_json=inventory.to_dict(),
            image_b64=_img_b64,                           # [추가]
        )

        print("\n===== PLAN (initial) =====")
        print(plan)
        print("=========================\n")

        # Guard 평가 및 필요 시 재계획
        MAX_GUARD_REPLANS = 5
        guard_tries = 0
        ok_guard, guard_msg = _guard_pseudo_function(pseudo_lines, world, inventory)
        while (not ok_guard) and (guard_tries < MAX_GUARD_REPLANS):
            print(f"[RUN] Guard 실패 → 재계획 시도 {guard_tries+1}\n{guard_msg}\n")
            # Guard 피드백을 planner에 그대로 전달
            plan, pseudo_lines = planner(
                Actor_Prompt=Actor_Prompt,
                user_instruction=user_instruction,
                world_json=world_json_dict,
                history_pseudo_function=history_blocks,
                critic_feedbacks=guard_msg,
                inventory_json=inventory.to_dict(),
                image_b64=_img_b64,                       # [추가]
            )
            print(f"\n===== PLAN (guard replan {guard_tries+1}) =====")
            print(plan)
            print("==============================================\n")
            ok_guard, guard_msg = _guard_pseudo_function(pseudo_lines, world, inventory)
            guard_tries += 1

        # Critic 평가 및 필요 시 재계획
        MAX_CRITIC_REPLANS = 5  # 무한루프 방지용
        replan_tries = 0

        critique = critic(
            Critic_Prompt=Critic_Prompt,
            user_instruction=user_instruction,
            world_json=world_json_dict,
            plan=plan,
            history_pseudo_function=history_blocks,
            inventory_json=inventory.to_dict(),
        )
        print(f"\n===== CRITIQUE =====")
        print(critique)
        print("==============================================\n")

        while needs_revision(critique) and (replan_tries < MAX_CRITIC_REPLANS):
            print(f"[RUN] Critic: REVISE REQUIRED → 재계획 시도 {replan_tries+1}")
            # Critic 피드백을 싣고 재계획
            plan, pseudo_lines = planner(
                Actor_Prompt=Actor_Prompt,
                user_instruction=user_instruction,
                world_json=world_json_dict,
                history_pseudo_function=history_blocks,
                critic_feedbacks=critique,
                inventory_json=inventory.to_dict(),
                image_b64=_img_b64,                       # [추가]
            )
            print(f"\n===== PLAN (replan try {replan_tries+1}) =====")
            print(plan)
            print("=============================================\n")

            # 다시 Guard 평가 및 필요 시 재계획
            guard_tries_crit = 0
            ok_guard, guard_msg = _guard_pseudo_function(pseudo_lines, world, inventory)
            while (not ok_guard) and (guard_tries_crit < MAX_GUARD_REPLANS):
                print(f"[RUN] Guard(critic 단계) 실패 → 재계획 시도 {guard_tries_crit+1}\n{guard_msg}\n")
                # Guard 피드백을 planner에 전달해서 다시 계획
                plan, pseudo_lines = planner(
                    Actor_Prompt=Actor_Prompt,
                    user_instruction=user_instruction,
                    world_json=world_json_dict,
                    history_pseudo_function=history_blocks,
                    critic_feedbacks=guard_msg,
                    inventory_json=inventory.to_dict(),
                    image_b64=_img_b64,                   # [추가]
                )
                print(f"\n===== PLAN (guard@critic replan {guard_tries_crit+1}) =====")
                print(plan)
                print("===========================================================\n")
                ok_guard, guard_msg = _guard_pseudo_function(pseudo_lines, world, inventory)
                guard_tries_crit += 1

            if not ok_guard:
                print("[RUN] Guard(critic 단계) 재계획 한도 도달 → 현재 계획으로 Critic 평가 진행")

            # 새 pseudo_function을 다시 Critic 평가
            critique = critic(
                Critic_Prompt=Critic_Prompt,
                user_instruction=user_instruction,
                world_json=world_json_dict,
                plan=plan,
                history_pseudo_function=history_blocks,
                inventory_json=inventory.to_dict(),
            )

            print(f"\n===== CRITIQUE =====")
            print(critique)
            print("==============================================\n")

            replan_tries += 1
            if needs_revision(critique) and (replan_tries >= MAX_CRITIC_REPLANS):
                print("[RUN] 재계획 한도 도달 → 현재 계획으로 진행")
                break

        # 4) search() 유무 확인
        has_search, search_line = locate_search(pseudo_lines)

        if has_search:
            executed = pseudo_lines[:search_line]  # search() 포함
            for ln in executed:
                _execute_pseudocode_lines([ln], world, ros_node)
                update_inventory_from_lines([ln], world, inventory)
            history_blocks.append("\n".join(pseudo_lines))
            print("[RUN] search()까지 실행 후 다음 루프로 재계획")
            continue

        # 5) search()가 없으면 전체 실행 → 미션 완료
        for ln in pseudo_lines:
            _execute_pseudocode_lines([ln], world, ros_node)
            update_inventory_from_lines([ln], world, inventory)
        print("[RUN] Mission complete.")
        break

    else:
        print("[RUN] slam_stream이 소진되어 종료되었습니다.")

# ─────────────────────────────────────────────────────────────
# ROS2 Node wrapper to feed slam_stream
# ─────────────────────────────────────────────────────────────
class DetectedObjectsNode(Node):
    """
    - test_mode=True: just read scenario JSON file and pass list to run()
    - test_mode=False: subscribe to 'detected_objects' (std_msgs/String with JSON)
      and provide a generator that yields parsed dicts to run()
    """
    def __init__(self):
        super().__init__("world_planner_node")

        # Parameters
        self.declare_parameter("test_mode", False)
        self.declare_parameter("scenario_path", "/home/nx/OFM/code/scenario_1.json")
        self.declare_parameter("detected_topic", "detected_objects_json")
        # self.declare_parameter("user_instruction", "Pick up the laundry and put it into the washing machine.")
        self.declare_parameter("user_instruction", "Pick up the dog and place it near the person.")
        self.declare_parameter("max_loops", 20)

        self.declare_parameter("target_pose_topic", "target_pose")
        self.declare_parameter("control_compl_topic", "control_compl")
        # self.declare_parameter("keywords_topic", "keywords")
        self.declare_parameter("keywords_topic", "typed_input")
        # [추가] 지도 입력 토픽과 형식( image | compressed | grid )
        self.declare_parameter("merged_map_topic", "merged_map")
        self.declare_parameter("merged_map_type", "image")  # "image" | "compressed" | "grid"
        self.declare_parameter("merged_map_qos", "sensor")  # "sensor" | "reliable"
        merged_map_qos = self.get_parameter("merged_map_qos").get_parameter_value().string_value

        # 프로파일 결정
        if merged_map_qos == "sensor":
            qos_profile = QoSPresetProfiles.SENSOR_DATA.value  # depth=5, BestEffort, Volatile
        else:
            qos_profile = QoSProfile(depth=5,
                                    reliability=QoSReliabilityPolicy.RELIABLE,
                                    durability=QoSDurabilityPolicy.VOLATILE)


        self.test_mode: bool = self.get_parameter("test_mode").get_parameter_value().bool_value
        self.scenario_path: str = self.get_parameter("scenario_path").get_parameter_value().string_value
        self.detected_topic: str = self.get_parameter("detected_topic").get_parameter_value().string_value
        self.user_instruction: str = self.get_parameter("user_instruction").get_parameter_value().string_value
        self.max_loops: int = int(self.get_parameter("max_loops").get_parameter_value().integer_value)
        self.target_pose_topic: str = self.get_parameter("target_pose_topic").get_parameter_value().string_value
        self.control_compl_topic: str = self.get_parameter("control_compl_topic").get_parameter_value().string_value
        self.keywords_topic: str = self.get_parameter("keywords_topic").get_parameter_value().string_value
        self.merged_map_topic: str = self.get_parameter("merged_map_topic").get_parameter_value().string_value
        self.merged_map_type: str = self.get_parameter("merged_map_type").get_parameter_value().string_value
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=100)
        self._sub = None

        # [추가] 최근 지도 이미지(base64) 저장용
        self.bridge = CvBridge()
        self.latest_image_b64: str | None = None

        if not self.test_mode:
            self._sub = self.create_subscription(
                StringMsg,
                self.detected_topic,
                self._detected_cb,
                10
            )
            self.get_logger().info(f"Subscribed to '{self.detected_topic}' for world JSON.")
        
        self.keywords_pub = self.create_publisher(StringMsg, self.keywords_topic, 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 10)

        # [추가] 지도 이미지 구독 (형식에 따라 다르게)
        
        if self.merged_map_type == "image":
            self.create_subscription(Image, self.merged_map_topic, self._map_image_cb, qos_profile)   # [변경]
            self.get_logger().info(f"Subscribed image: '{self.merged_map_topic}' (QoS: {merged_map_qos})")

        elif self.merged_map_type == "compressed":
            self.create_subscription(CompressedImage, self.merged_map_topic, self._map_compressed_cb, qos_profile)  # [변경]
            self.get_logger().info(f"Subscribed compressed: '{self.merged_map_topic}' (QoS: {merged_map_qos})")

        elif self.merged_map_type == "grid":
            # grid도 퍼블리셔가 sensor QoS일 수 있으니 동일 프로파일 사용 (필요시 reliable로 바꿔 사용)
            self.create_subscription(OccupancyGrid, self.merged_map_topic, self._map_grid_cb, qos_profile)  # [변경]
            self.get_logger().info(f"Subscribed grid: '{self.merged_map_topic}' (QoS: {merged_map_qos})")

            
        self._control_event = threading.Event()
        self._control_done = False
        self.control_sub = self.create_subscription(
            BoolMsg,
            self.control_compl_topic,
            self._control_cb,
            10
        )

# ndarray(BGR/GRAY) -> PNG -> base64
    def _encode_ndarray_to_b64(self, img_nd: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", img_nd)
        if not ok:
            raise RuntimeError("cv2.imencode('.png', ...) failed")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # sensor_msgs/Image 콜백
    def _map_image_cb(self, msg: Image):
        try:
            cvimg = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image_b64 = self._encode_ndarray_to_b64(cvimg)
        except Exception as e:
            self.get_logger().warn(f"map image convert fail: {e}")

    # sensor_msgs/CompressedImage 콜백
    def _map_compressed_cb(self, msg: CompressedImage):
        try:
            nparr = np.frombuffer(msg.data, np.uint8)
            cvimg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.latest_image_b64 = self._encode_ndarray_to_b64(cvimg)
        except Exception as e:
            self.get_logger().warn(f"map compressed convert fail: {e}")

    # nav_msgs/OccupancyGrid 콜백
    def _map_grid_cb(self, msg: OccupancyGrid):
        try:
            w = msg.info.width
            h = msg.info.height
            data = np.array(msg.data, dtype=np.int16).reshape(h, w)  # -1, 0~100
            img = np.full((h, w), 128, dtype=np.uint8)               # unknown=128
            known = data >= 0
            img[known] = (255 - (data[known] * 255 // 100)).astype(np.uint8)  # 0=자유->255, 100=점유->0
            self.latest_image_b64 = self._encode_ndarray_to_b64(img)
        except Exception as e:
            self.get_logger().warn(f"occupancy grid convert fail: {e}")
            
    def _detected_cb(self, msg: StringMsg):
        try:
            data = json.loads(msg.data)
            # Non-blocking put; if full, drop the oldest to keep freshest
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(data)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse detected_objects JSON: {e}")

    def _control_cb(self, msg: BoolMsg):
        if bool(msg.data):
            self._control_done = True
            self._control_event.set()
            self.get_logger().info("control_compl=True 수신")
        else:
            self._control_done = False
            self._control_event.clear()

    def publish_keywords(self, text: str):
        self.keywords_pub.publish(StringMsg(data=str(text)))
        self.get_logger().info(f"keywords -> '{self.keywords_topic}': '{text}'")

    def publish_goal(self, x: float, y: float, yaw: float = 0.0):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(yaw/2.0)
        msg.pose.orientation.w = math.cos(yaw/2.0)
        self.target_pose_pub.publish(msg)
        self.get_logger().info(
            f"target_pose -> '{self.target_pose_topic}': x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}"
        )

    def wait_for_control_complete(self, timeout: Optional[float] = None) -> bool:
        return self._control_event.wait(timeout=timeout)

    def reset_control_complete(self):
        self._control_done = False
        self._control_event.clear()

    def message_generator(self):
        """
        Blocking generator that yields parsed JSON dicts from the subscription queue.
        This matches the slam_stream interface expected by run().
        """
        while rclpy.ok():
            try:
                item = self._queue.get(timeout=0.5)
                yield item
            except queue.Empty:
                continue

# ─────────────────────────────────────────────────────────────
# main: ROS2 entrypoint while preserving original run() flow
# ─────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = DetectedObjectsNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # test_mode가 초기 False이면, 큰따옴표로 감싼 문장 입력 시 test_mode=True로 전환
        if not node.test_mode:
            try:
                raw = input('실행할 문장을 큰따옴표("")안에 입력: ').strip()
            except EOFError:
                raw = ""
            m = re.match(r'^\s*"(.*)"\s*$', raw)
            if m:
                node.user_instruction = m.group(1)
                node.test_mode = True
                node.get_logger().info(f"입력 수신 → test_mode=True 전환, user_instruction='{node.user_instruction}'")
            else:
                node.get_logger().info("입력 미수신 또는 형식 불일치 → live 모드 유지")

        # TEST MODE: 파일 기반
        if node.test_mode:
            scenario_path = node.scenario_path
            with open(scenario_path, "r", encoding="utf-8") as f:
                slam_stream = json.load(f)
            run(node.user_instruction, slam_stream, max_loops=node.max_loops, ros_node=node)
        else:
            # LIVE MODE: detected_objects 구독
            run(node.user_instruction, node.message_generator(), max_loops=node.max_loops, ros_node=node)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()



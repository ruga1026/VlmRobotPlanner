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
- If the generated plan contains search(...), the node executes lines up to (and including) search()
  and then WAITS for terminal input:
    * any input → publish the search object (string, no quotes) on planner/search_signal
    * "end"     → publish "X" on planner/search_signal and RESUME planning
- If there’s no search(...) in the plan, it executes all steps and marks mission complete.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List, Optional
import json, re, ollama, time, threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

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
deleted for now
"""

Critic_Prompt = """
deleted for now
"""

PLANNER_MODEL = "qwen2.5vl:7b"   # VLM (planner 용)
CRITIC_MODEL  = "qwen3:1.7b"     # LLM (critic 용)

TEMPERATURE_PLANNER = 0.0
TEMPERATURE_CRITIC  = 0.0

def call_ollama(messages, model=str, temperature: float = 0.2):
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
        model=PLANNER_MODEL,
        temperature=TEMPERATURE_PLANNER,
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

def _execute_pseudocode_lines(lines: list[str], delay_sec: float = 5.0, logger=lambda s: None):
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
                    f"pick target '{arg}' is not present in the current world_json. Consider calling search({arg} first. "
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

        # Pull params into module-level globals the original functions use
        PLANNER_MODEL = self.get_parameter("planner_model").get_parameter_value().string_value
        CRITIC_MODEL  = self.get_parameter("critic_model").get_parameter_value().string_value
        TEMPERATURE_PLANNER = float(self.get_parameter("temperature_planner").value)
        TEMPERATURE_CRITIC  = float(self.get_parameter("temperature_critic").value)

        self.user_instruction = self.get_parameter("user_instruction").get_parameter_value().string_value
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
        self.sub_slam = self.create_subscription(String, "slam/json", self._on_slam_msg, 10)

        # State
        self.world = WorldModel()
        self.inventory = InventoryModel()
        self.history_blocks: list[str] = []
        self.loop_idx = 0
        self.mission_complete = False
        self._busy_lock = threading.Lock()

        # NEW: terminal watcher thread state + wait gate
        self._search_sig_thread: Optional[threading.Thread] = None
        self._waiting_for_search_input: bool = False   # ← pause gate while search() is active

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

        self._emit_status("Planner node ready.")

    # ─────────────────────────────
    # Helpers to publish logs/status
    # ─────────────────────────────
    def _emit_exec(self, msg: str):
        self.get_logger().info(msg)
        self.pub_exec_log.publish(String(data=msg))

    def _emit_status(self, msg: str):
        self.get_logger().info(msg)
        self.pub_status.publish(String(data=msg))

    # NEW: background terminal watcher for search(object)
    def _start_search_signal(self, obj_name: Optional[str]) -> None:
        # Only one watcher at a time
        if self._search_sig_thread and self._search_sig_thread.is_alive():
            self._emit_exec("[SEARCH] signal thread already running; skipping re-start")
            return

        # Engage the wait gate
        self._waiting_for_search_input = True

        def _worker():
            try:
                pretty = obj_name or ""
                self._emit_status(
                    f"[SEARCH] Planned search('{pretty}') passed guard/critic. "
                    f"Type anything to publish '{pretty}', or 'end' to publish 'X' and resume."
                )
                while True:
                    try:
                        line = input()
                    except EOFError:
                        # No TTY available (e.g., launched without stdin) → publish 'X' and exit
                        self.pub_search_signal.publish(String(data="X"))
                        self._emit_status("[SEARCH] No stdin (EOF). Published 'X' and exiting watcher.")
                        break
                    if line.strip().lower() == "end":
                        self.pub_search_signal.publish(String(data="X"))
                        self._emit_status("[SEARCH] Received 'end' → published 'X'.")
                        break
                    # any other input → publish the object name (no quotes) and keep waiting
                    self.pub_search_signal.publish(String(data=str(pretty)))
                    self._emit_status(f"[SEARCH] Published '{pretty}' on planner/search_signal.")
            except Exception as e:
                self._emit_status(f"[SEARCH] watcher error: {e}")
            finally:
                # Disengage the wait gate so planning can resume
                self._waiting_for_search_input = False
                self._emit_status("[SEARCH] Input session ended; resuming planning.")

        self._search_sig_thread = threading.Thread(target=_worker, daemon=True)
        self._search_sig_thread.start()

    # Scenario timer — feeds frames into the same processing path
    def _scenario_tick_cb(self):
        # PAUSE scenario replay while waiting for search input
        if self._waiting_for_search_input:
            self._emit_exec("[RUN] Waiting for search input; pausing scenario tick.")
            return

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
        if self.mission_complete:
            return

        # Drop SLAM frames while waiting for terminal input from search()
        if self._waiting_for_search_input:
            self._emit_exec("[RUN] Waiting for terminal input ('search' in progress); skipping SLAM processing.")
            return

        if self.loop_idx >= self.max_loops:
            self._emit_status("최대 루프 수에 도달했습니다. 미션 실패.")
            return
        if not self._busy_lock.acquire(blocking=False):
            # Drop frame if still processing previous one (keeps logic simple)
            self._emit_exec("[RUN] Busy; skipping incoming SLAM frame.")
            return

        try:
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
                executed = pseudo_lines[:search_line]  # includes the search() line itself
                for ln in executed:
                    _execute_pseudocode_lines([ln], delay_sec=self.exec_delay_sec, logger=self._emit_exec)
                    update_inventory_from_lines([ln], self.world, self.inventory, logger=self._emit_exec)

                # NEW: extract search target and start terminal watcher
                try:
                    _, arg = _parse_call(pseudo_lines[search_line-1])
                except Exception:
                    arg = None
                self._start_search_signal(arg)

                self.history_blocks.append("\n".join(pseudo_lines))
                self.pub_inventory_json.publish(String(data=self.inventory.to_json(indent=2, ensure_ascii=False)))
                self._emit_status("[RUN] search()까지 실행 완료 — 터미널 입력 대기 중 (scenario/SLAM 일시 정지).")
                return  # wait gate engaged; resume after 'end' input publishes "X"

            # 5) search()가 없으면 전체 실행 → 미션 완료
            for ln in pseudo_lines:
                _execute_pseudocode_lines([ln], delay_sec=self.exec_delay_sec, logger=self._emit_exec)
                update_inventory_from_lines([ln], self.world, self.inventory, logger=self._emit_exec)

            self.history_blocks.append("\n".join(pseudo_lines))
            self.pub_inventory_json.publish(String(data=self.inventory.to_json(indent=2, ensure_ascii=False)))
            self._emit_status("[RUN] Mission complete.")
            self.mission_complete = True

        finally:
            self._busy_lock.release()

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

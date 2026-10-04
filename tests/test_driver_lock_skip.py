"""自动播放驱动不被单会话长持锁拖停的回归测试。

修复前：后台驱动线程（以及确定性的 tick_autoplay）逐个会话以**阻塞**方式
进入会话锁。先创建的未启动会话保存存档时，会话锁被保存持有、保存本身在
锁内等待 SQLite 写锁，驱动便挂在该会话的锁上——同拍之后（包括后创建的）
运行会话的游标推进与事件入队全部停摆。

修复后每拍对各会话**非阻塞**进锁（acquire(blocking=False)）：

- 被占用的会话本拍跳过，下一拍（50ms 后）再试；未被占用的运行会话照常
  每拍推进，不被其他会话的持锁拖停；
- 被占用会话与其自身操作仍严格串行（后台从不在锁外改动会话状态）；
- 锁释放后若会话仍 running，下一拍按**原计时段**（base_cursor_ms +
  自 base_monotonic_s 经过的整毫秒）一次性补齐游标与事件（事件窗口、
  排序、连续 seq、共享容量与整批阻塞规则不变）；
- 锁占用期间会话已被暂停或已结束的，解锁后不再推进。

下面用可控单调时钟与确定性的线程交错（持锁栅栏 + 排队屏障）分别覆盖：
1) 存档等 SQLite 写锁（保存持会话锁长等）拖停其他运行会话；
2) 运行会话接纳持锁期间其他会话照常推进，解锁后按原计时段补齐事件
   （连续 seq、排序）及满额整批阻塞；
3) 占用期间被暂停 / 被结束的会话，解锁后不得再推进；
4) HTTP + 真实 SQLite 写锁的端到端冒烟：保存卡住时后创建的运行会话
   仍按真实时间推进，写锁释放后保存成功。
"""
import os
import sqlite3
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import main as main_module  # noqa: E402
from app.main import app  # noqa: E402
from app.playback import (  # noqa: E402
    AUTOPLAY_BLOCKED,
    AUTOPLAY_PAUSED,
    AUTOPLAY_RUNNING,
    PlaybackManager,
)
from app.storage import SceneRecord  # noqa: E402


# ---------------------------------------------------------------------------
# 可控时钟与确定性持锁栅栏
# ---------------------------------------------------------------------------


class FakeClock:
    """可控单调时钟：按毫秒步进，由管理器在会话锁内读取。"""

    def __init__(self, now_s=0.0):
        self._now_s = now_s
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._now_s

    def advance_ms(self, ms):
        with self._lock:
            self._now_s += ms / 1000.0


class HeldLockGate:
    """可确定性地长时间阻塞后台线程的栅栏。

    主线程在构造时即持有 gate 锁（acquire）；工作线程调用 block() 时会
    一直阻塞在 gate 锁上，直到主线程调用 release()——Event.wait 可能被
    超时/虚假唤醒打断，已持有的互斥锁不会。用于让“持会话锁的工作线程”
    在测试选定的整个区间内稳定地不释放会话锁。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._lock.acquire()

    def block(self, timeout=10):
        got = self._lock.acquire(timeout=timeout)
        if not got:  # pragma: no cover - 测试缺陷
            raise RuntimeError("HeldLockGate 未在超时内放行")
        # 拿到后立即归还，保持锁处于“释放后可用”的常规状态
        self._lock.release()

    def release(self):
        self._lock.release()


class QueuedBarrier:
    """等待一个控制请求确定地排到会话锁上（在锁外、未持锁）。

    工作线程在“已取得会话引用、即将 with session.lock”处调用 wait_then()
    ：先到屏障（parties=2），主线程放行后它才去抢会话锁；由于会话锁此
    刻被另一个工作线程持有，控制请求确定地在锁外排队，且尚未读取/改动
    任何会话状态。
    """

    def __init__(self):
        self._barrier = threading.Barrier(2)

    def wait_then(self, fn):
        self._barrier.wait(timeout=10)
        return fn()

    def wait_queued(self):
        self._barrier.wait(timeout=10)


class BlockingPersist:
    """模拟存档落库在会话锁内长时间阻塞（等待 SQLite 写锁）。

    commit_checkpoint 在持有会话锁时调用 persist：进入后置位 entered，
    随后阻塞在 held gate 上直到主线程放行（放行后默认成功，不真正写库）。
    """

    def __init__(self):
        self.entered = threading.Event()
        self.gate = HeldLockGate()
        self.calls = 0

    def __call__(self, payload):
        self.calls += 1
        self.entered.set()
        self.gate.block()


class AcceptHoldingLock:
    """在会话锁内确定地阻塞的接纳：模拟运行会话接纳持锁。

    accept 的 load_record 回调在会话锁内执行：进入后置位 entered 并阻塞
    在 held gate 上，直到上下文退出才放行（返回给定快照，接纳成功）。
    持锁期间该会话的锁被稳定占用，驱动每拍非阻塞尝试都会跳过它。
    """

    def __init__(self, pm, session, record_to_return=None):
        self._pm = pm
        self._session = session
        self._record = record_to_return
        self.entered = threading.Event()
        self.gate = HeldLockGate()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error = None
        self.result = None

    def _load(self):
        self.entered.set()
        self.gate.block()
        return self._record

    def _run(self):
        try:
            self.result = self._pm.accept(self._session, self._load)
        except Exception as exc:  # pragma: no cover - 失败即测试缺陷
            self.error = exc

    def __enter__(self):
        self.thread.start()
        assert self.entered.wait(timeout=10)
        return self

    def __exit__(self, *exc_info):
        self.gate.release()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()
        assert self.error is None
        return False


def _record(scene_id, subtitles, tracks=None, track_count=2):
    subs = [dict(s) for s in subtitles]
    if tracks is None:
        # 与测试场景一致地交替排轨，避免重叠冲突
        tracks = [i % track_count + 1 for i in range(len(subs))]
    return SceneRecord(
        scene_id=scene_id,
        revision=1,
        track_count=track_count,
        subtitles=subs,
        assignments=[
            {"id": s["id"], "track": t} for s, t in zip(subs, tracks)
        ],
        changes=0,
    )


def _open_session(pm, record):
    state = pm.create(record)
    session = pm.get(state["session_id"])
    pm.open_channel(session)
    return session


def _clock_manager(clock, limit=64):
    return PlaybackManager(
        channel_event_limit=limit,
        autoplay_tick_interval_s=None,
        clock=clock,
    )


def _queued(pm, session, ack=0):
    return pm.fetch_channel(session, ack)["events"]


# 两个标准场景：
# A 场景事件稀疏（x[100,200)、y[300,400)），B 场景只有一个 b[150,5000)。
A_SUBS = [
    {"id": "x", "speaker": "X", "start_ms": 100, "end_ms": 200},
    {"id": "y", "speaker": "Y", "start_ms": 300, "end_ms": 400},
]
B_SUBS = [
    {"id": "b", "speaker": "B", "start_ms": 150, "end_ms": 5000},
]


# ---------------------------------------------------------------------------
# 1) 存档等（模拟的）写锁：驱动不挂在被占用会话上，其他运行会话照常推进
# ---------------------------------------------------------------------------


def test_checkpoint_persist_wait_does_not_stall_other_running_sessions():
    clock = FakeClock()
    pm = _clock_manager(clock)

    # 先创建、已开通道但**未启动**自动播放的会话 A：对它存档合法
    a = _open_session(pm, _record("scene-a-cp", A_SUBS, tracks=[1, 2]))
    # 后创建的运行会话 B
    b = _open_session(pm, _record("scene-b-cp", B_SUBS, tracks=[1]))
    pm.start_autoplay(b, now=0.0)

    gate = BlockingPersist()
    outcomes = {}

    def do_save():
        outcomes["payload"] = pm.commit_checkpoint(a, gate)

    saver = threading.Thread(target=do_save, daemon=True)
    saver.start()
    try:
        assert gate.entered.wait(timeout=10)
        # 保存此刻持有 A 的会话锁并阻塞在“写库”上。时钟推进 300ms，
        # 连续多拍：A 被跳过，B 每拍都按原计时段推进，绝不被拖停。
        clock.advance_ms(100)
        pm.tick_autoplay()
        assert a.cursor_ms == 0  # idle 会话永远不动
        assert b.cursor_ms == 100
        assert [(e["seq"], e["id"], e["type"]) for e in _queued(pm, b)] == []

        clock.advance_ms(100)
        pm.tick_autoplay()
        assert b.cursor_ms == 200
        assert [(e["seq"], e["id"], e["type"]) for e in _queued(pm, b)] == [
            (1, "b", "enter")  # b@150 在 (100,200] 窗口内入队
        ]

        clock.advance_ms(100)
        pm.tick_autoplay()
        assert b.cursor_ms == 300
        # 被占用的 A 保持 idle，保存尚未“提交”
        assert a.autoplay_status == "idle"
        assert gate.calls == 1
        # 多拍期间锁一直被保存线程持有（非阻塞跳过，不去抢它）
        assert saver.is_alive()
    finally:
        gate.gate.release()
        saver.join(timeout=10)
    assert not saver.is_alive()

    # 保存成功返回，序列化的是持锁时的 A 状态（游标 0、空队列）
    assert outcomes["payload"]["cursor_ms"] == 0
    assert outcomes["payload"]["autoplay_status"] == "idle"
    # B 不受影响，继续推进
    clock.advance_ms(100)
    pm.tick_autoplay()
    assert b.cursor_ms == 400


# ---------------------------------------------------------------------------
# 2) 运行会话接纳持锁：其他会话照常推进；解锁后按原计时段补齐事件
# ---------------------------------------------------------------------------


def test_running_session_accept_holds_lock_others_advance_and_it_catches_up():
    clock = FakeClock()
    pm = _clock_manager(clock)
    a_record = _record("scene-a-accept", A_SUBS, tracks=[1, 2])
    a = _open_session(pm, a_record)
    b = _open_session(pm, _record("scene-b-accept", B_SUBS, tracks=[1]))
    pm.start_autoplay(a, now=0.0)
    pm.start_autoplay(b, now=0.0)

    with AcceptHoldingLock(pm, a, record_to_return=a_record):
        # 接纳稳定持有 A 锁。时钟走 300ms，多拍：A 被跳过（游标不动、
        # 不消耗 seq），B 每拍照常推进。
        for ms, b_cursor in ((100, 100), (200, 200), (300, 300)):
            clock.advance_ms(100)
            pm.tick_autoplay()
            assert a.cursor_ms == 0
            assert b.cursor_ms == b_cursor
        assert a.channel.next_seq == 1  # 跳过期间未消耗任何 seq

    # 接纳完成（返回相同快照：成功、仍 running、计时段起点不变）
    assert a.autoplay_status == AUTOPLAY_RUNNING
    assert a.base_monotonic_s == 0.0
    assert a.cursor_ms == 0

    # 解锁后下一拍：仍 running，按**原计时段**一次性补齐到 350——
    # 窗口 (0,350] 内三条事件按既有排序（时刻升序、同刻先 exit 后 enter）
    # 以连续 seq 整批入队，没有因被跳过而漏事件。
    clock.advance_ms(50)
    pm.tick_autoplay()
    assert a.cursor_ms == 350
    assert [(e["seq"], e["time_ms"], e["type"], e["id"]) for e in _queued(pm, a)] == [
        (1, 100, "enter", "x"),
        (2, 200, "exit", "x"),
        (3, 300, "enter", "y"),
    ]
    # B 同拍继续推进
    assert b.cursor_ms == 350
    assert [(e["seq"], e["id"], e["type"]) for e in _queued(pm, b)] == [
        (1, "b", "enter")
    ]


def test_catch_up_batch_overflow_goes_blocked_whole_batch_then_resume():
    """解锁后补齐也是整批：容量不足整批转 blocked，游标/seq/队列不变。"""
    clock = FakeClock()
    pm = _clock_manager(clock, limit=2)
    a_record = _record("scene-a-overflow", A_SUBS, tracks=[1, 2])
    a = _open_session(pm, a_record)
    pm.start_autoplay(a, now=0.0)

    with AcceptHoldingLock(pm, a, record_to_return=a_record):
        clock.advance_ms(300)
        pm.tick_autoplay()  # A 被跳过
        assert a.cursor_ms == 0

    # 解锁后拍到 300：补齐整批 3 条事件，容量仅 2 -> blocked，
    # 游标、seq、队列全部保持原样（不部分入队）。
    pm.tick_autoplay()
    assert a.autoplay_status == AUTOPLAY_BLOCKED
    assert a.cursor_ms == 0
    assert _queued(pm, a) == []
    assert a.channel.next_seq == 1
    assert pm.channel_capacity.used == 0

    # 显式恢复后以保留游标 0 重新计时（不追赶），+125ms（二进制可精确
    # 表示的秒数）的拍只产生 x@100 enter 一条，容量充足 -> running
    pm.resume_autoplay(a, now=10.0)
    pm.tick_autoplay(now=10.125)
    assert a.autoplay_status == AUTOPLAY_RUNNING
    assert a.cursor_ms == 125
    assert [(e["seq"], e["id"], e["type"]) for e in _queued(pm, a)] == [
        (1, "x", "enter")
    ]


# ---------------------------------------------------------------------------
# 3) 占用期间被暂停 / 被结束：解锁后不得再推进
# ---------------------------------------------------------------------------


def _queue_control_while_lock_held(make_call):
    """启动一个控制请求线程，并确定地等到它排在会话锁外。

    make_call 为实际调用（在屏障放行后才执行）；返回已启动的线程。
    会话锁此刻必须正被另一个工作线程持有，因此控制请求确定地排队、
    尚未读取/改动任何会话状态。
    """
    barrier = QueuedBarrier()
    thread = threading.Thread(
        target=lambda: barrier.wait_then(make_call), daemon=True
    )
    thread.start()
    barrier.wait_queued()
    return thread


def test_session_paused_while_lock_held_is_not_advanced_after_unlock():
    clock = FakeClock()
    pm = _clock_manager(clock)
    a_record = _record("scene-a-pause", A_SUBS, tracks=[1, 2])
    a = _open_session(pm, a_record)
    b = _open_session(pm, _record("scene-b-pause", B_SUBS, tracks=[1]))
    pm.start_autoplay(a, now=0.0)
    pm.start_autoplay(b, now=0.0)

    paused_result = {}
    with AcceptHoldingLock(pm, a, record_to_return=a_record):
        # 暂停请求在会话锁外确定地排队（接纳先持锁）
        pause_thread = _queue_control_while_lock_held(
            lambda: paused_result.update(pm.pause_autoplay(a))
        )
        clock.advance_ms(300)
        pm.tick_autoplay()  # A 锁被占用：跳过；B 照常推进
        assert a.cursor_ms == 0
        assert b.cursor_ms == 300
        assert pause_thread.is_alive()  # 暂停仍在等锁

    # 接纳先完成、暂停随后生效
    pause_thread.join(timeout=10)
    assert paused_result["status"] == AUTOPLAY_PAUSED
    assert a.cursor_ms == 0

    # 解锁后再走多久都不推进已暂停的 A；B 不受影响
    clock.advance_ms(5000)
    pm.tick_autoplay()
    assert a.autoplay_status == AUTOPLAY_PAUSED
    assert a.cursor_ms == 0
    assert _queued(pm, a) == []
    assert b.cursor_ms >= 5000


def test_session_finished_while_lock_held_is_not_advanced_after_unlock():
    clock = FakeClock()
    pm = _clock_manager(clock)
    a_record = _record("scene-a-finish", A_SUBS, tracks=[1, 2])
    a = _open_session(pm, a_record)
    b = _open_session(pm, _record("scene-b-finish", B_SUBS, tracks=[1]))
    pm.start_autoplay(a, now=0.0)
    pm.start_autoplay(b, now=0.0)
    a_id = a.session_id

    finish_result = {}
    finish_error = []

    def do_finish():
        finish_result["state"] = pm.finish(a_id)

    with AcceptHoldingLock(pm, a, record_to_return=a_record):
        # 结束请求在注册表查到会话后，于会话锁外确定地排队
        finish_thread = _queue_control_while_lock_held(do_finish)
        clock.advance_ms(300)
        pm.tick_autoplay()  # A 被跳过；B 照常推进
        assert a.cursor_ms == 0
        assert b.cursor_ms == 300
        assert finish_thread.is_alive()  # 结束仍在等锁

    # 接纳先完成、结束随后生效（一次成功）
    finish_thread.join(timeout=10)
    assert finish_error == []
    assert finish_result["state"]["session_id"] == a_id
    assert pm.get(a_id) is None  # 注册表已移除

    # 解锁后的任何拍都不再推进 A（注册表里已没有它），B 继续
    clock.advance_ms(100)
    pm.tick_autoplay()
    assert b.cursor_ms == 400


# ---------------------------------------------------------------------------
# 4) HTTP + 真实 SQLite 写锁：保存卡住时后创建的运行会话仍按墙钟推进
# ---------------------------------------------------------------------------


def _wait_for(condition, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise AssertionError("条件在超时时间内未满足")


def _http_session(client, scene_id, subtitles, tracks=None):
    track_count = 2
    if tracks is None:
        tracks = [i % track_count + 1 for i in range(len(subtitles))]
    put = client.put(
        f"/scenes/{scene_id}",
        json={
            "track_count": track_count,
            "expected_revision": 0,
            "subtitles": subtitles,
        },
    )
    assert put.status_code == 200
    resp = client.post("/sessions", json={"scene_id": scene_id})
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    assert client.post(f"/sessions/{sid}/channel").status_code == 200
    return sid


def test_http_checkpoint_waiting_real_sqlite_write_lock_keeps_other_autoplay_running():
    client = TestClient(app)
    # 先创建的会话 A：开通道、不启动自动播放（存档合法）
    sid_a = _http_session(
        client,
        "scene-stall-a",
        [{"id": "a1", "speaker": "A", "start_ms": 0, "end_ms": 60000}],
        tracks=[1],
    )
    # 后创建的会话 B：场景与通道在取写锁之前就绪（写锁会阻塞一切写入，
    # 包括创建场景），取锁后只需启动自动播放这一纯内存操作。
    sid_b = _http_session(
        client,
        "scene-stall-b",
        [{"id": "b1", "speaker": "B", "start_ms": 0, "end_ms": 60000}],
        tracks=[1],
    )

    # 另一个写入者持有 SQLite 写锁（与保存的 BEGIN IMMEDIATE 互斥）
    db_path = os.environ["SCENES_DB_PATH"]
    lock_conn = sqlite3.connect(db_path)
    lock_conn.execute("BEGIN IMMEDIATE")

    save_resp = {}

    def do_save():
        c = TestClient(app, raise_server_exceptions=False)
        save_resp["r"] = c.put(
            f"/sessions/{sid_a}/checkpoints/cp-http-stall"
        )

    saver = threading.Thread(target=do_save, daemon=True)
    saver.start()
    try:
        # 给保存请求时间进入 SQLite 写锁等待（busy_timeout=30s，会一直等）
        time.sleep(0.8)
        assert saver.is_alive(), "保存应仍在等待 SQLite 写锁"
        # 保存尚未提交
        assert (
            client.post("/checkpoints/cp-http-stall/restore").status_code == 404
        )

        # B 启动自动播放是纯内存操作，不经过 SQLite
        started = client.post(f"/sessions/{sid_b}/autoplay/start")
        assert started.status_code == 200
        assert started.json()["cursor_ms"] == 0

        # 即便保存线程正持有 A 的会话锁并卡在写锁上，后台驱动也必须
        # 每 50ms 推进 B（修复前驱动整体挂在 A 的锁上）。
        _wait_for(
            lambda: client.get(f"/sessions/{sid_b}/autoplay").json()[
                "cursor_ms"
            ]
            >= 250,
            timeout=6.0,
        )
        body = client.get(f"/sessions/{sid_b}/autoplay").json()
        assert body["status"] == "running"
        assert body["cursor_ms"] >= 250

        # A 仍是 idle、保存仍卡着。注意：直接查询 A 的会话状态会阻塞在
        # A 的会话锁上（保存正持锁等写锁），从而把主线程的写锁回滚拖过
        # 30s busy_timeout——因此把该查询放到独立线程并在回滚后汇合：它
        # 串行等到保存完成后返回，状态必须仍是 idle、游标 0。
        def query_a():
            a_state["body"] = client.get(f"/sessions/{sid_a}/autoplay").json()

        a_state = {}
        query_a_thread = threading.Thread(target=query_a, daemon=True)
        query_a_thread.start()
        time.sleep(0.1)  # 让查询确定地排在 A 的会话锁外
        assert saver.is_alive()
    finally:
        lock_conn.rollback()
        lock_conn.close()
        saver.join(timeout=10)
        query_a_thread.join(timeout=10)
    assert not saver.is_alive()
    assert a_state["body"]["status"] == "idle"
    assert a_state["body"]["cursor_ms"] == 0

    # 写锁释放后保存成功
    assert save_resp["r"].status_code == 200
    assert save_resp["r"].json()["checkpoint_id"] == "cp-http-stall"
    assert save_resp["r"].json()["status"] == "idle"
    # 存档可恢复，A 不受影响
    resp = client.post("/checkpoints/cp-http-stall/restore")
    assert resp.status_code == 200
    assert resp.json()["cursor_ms"] == 0
    # B 在保存完成后继续自动推进
    before = client.get(f"/sessions/{sid_b}/autoplay").json()["cursor_ms"]
    _wait_for(
        lambda: client.get(f"/sessions/{sid_b}/autoplay").json()[
            "cursor_ms"
        ]
        > before
    )
    client.post(f"/sessions/{sid_b}/finish")

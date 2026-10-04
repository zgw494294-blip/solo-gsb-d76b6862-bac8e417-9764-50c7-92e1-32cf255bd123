"""定位（seek）：前跳或倒带到指定时刻，同步切换显示字幕。

覆盖：
- 基本定位：前跳/倒带/原位/零点/越过末尾；跳过区间不补发事件；原可见
  active 逐条 exit、目标 active 逐条 enter（两边都活动者也退出再进入）；
  事件沿用原字段、均标目标时刻，先 exit 后 enter、同类按会话字幕顺序；
- 响应含游标、源版本、可见 active、显示事件与自动播放状态；
- 隐藏集合参与两端可见性判断；定位不改快照、隐藏集合与接纳规则；
- 通道：未确认队列与确认边界保留，新事件接续 seq，时间回退仍按 seq
  补取；未开通道的定位不产生容量占用；
- 容量不足 503：游标、队列、seq、计时段与播放状态均不变，释放后原
  请求可重试（含运行中会话的确定性回归）；
- 自动播放：idle 保持 idle；running/paused/blocked 定位后为 paused；
  定位返回后旧驱动不再推进；resume 从新游标重新计时、不追赶；
- 404：未知或已结束会话；422：负数/布尔/字符串/浮点/缺失；
- 串行：并发定位完整串行，响应无中间态；结束后定位 404；
- 原推进仍禁止倒退；旧接口响应字段不变。
"""
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import main as main_module  # noqa: E402
from app.main import app  # noqa: E402
from app.playback import (  # noqa: E402
    AUTOPLAY_BLOCKED,
    AUTOPLAY_RUNNING,
    ChannelCapacityExhausted,
    PlaybackManager,
)
from app.storage import SceneRecord  # noqa: E402


def scene_payload(track_count=2):
    """s1[0,1000) s2[800,3000)锁2 s3[1000,2500) —— 分轨 1/2/1，换轨 0。"""
    return {
        "track_count": track_count,
        "expected_revision": 0,
        "subtitles": [
            {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
            {"id": "s2", "speaker": "Bob", "start_ms": 800, "end_ms": 3000,
             "locked_track": 2},
            {"id": "s3", "speaker": "Alice", "start_ms": 1000, "end_ms": 2500},
        ],
    }


def make_session(client, scene_id="scene-seek"):
    resp = client.put(f"/scenes/{scene_id}", json=scene_payload())
    assert resp.status_code == 200
    resp = client.post("/sessions", json={"scene_id": scene_id})
    assert resp.status_code == 200
    return resp.json()["session_id"]


def seek(client, sid, time_ms):
    return client.post(f"/sessions/{sid}/seek", json={"time_ms": time_ms})


def open_channel(client, sid):
    return client.post(f"/sessions/{sid}/channel")


def fetch(client, sid, ack_seq):
    return client.post(
        f"/sessions/{sid}/channel/fetch", json={"ack_seq": ack_seq}
    )


def autoplay_status(client, sid):
    return client.get(f"/sessions/{sid}/autoplay").json()


def _triples(body):
    return [(e["time_ms"], e["type"], e["id"]) for e in body["events"]]


def wait_for(condition, timeout=3.0, interval=0.01):
    """轮询直到 condition() 为真；超时断言失败。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise AssertionError("条件在超时时间内未满足")


# ---------------------------------------------------------------------------
# 基本定位：前跳 / 倒带 / 跳过区间不补发 / 事件字段与排序
# ---------------------------------------------------------------------------


def test_seek_forward_exits_old_visible_then_enters_target():
    """前跳：原可见 active 逐条 exit、目标 active 逐条 enter，均标目标时刻。"""
    client = TestClient(app)
    sid = make_session(client)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    resp = seek(client, sid, 1500)
    assert resp.status_code == 200
    body = resp.json()
    assert body["cursor_ms"] == 1500
    assert body["source_revision"] == 1
    # 900 处可见 s1、s2；1500 处可见 s2、s3（s2 两边都活动：退出再进入）
    assert body["events"] == [
        {"id": "s1", "speaker": "Alice", "track": 1,
         "time_ms": 1500, "type": "exit"},
        {"id": "s2", "speaker": "Bob", "track": 2,
         "time_ms": 1500, "type": "exit"},
        {"id": "s2", "speaker": "Bob", "track": 2,
         "time_ms": 1500, "type": "enter"},
        {"id": "s3", "speaker": "Alice", "track": 1,
         "time_ms": 1500, "type": "enter"},
    ]
    assert [(a["id"], a["track"]) for a in body["active"]] == [
        ("s2", 2), ("s3", 1),
    ]
    # 未启动自动播放的会话定位后仍为 idle
    assert body["status"] == "idle"
    assert body["autoplay_status"] == "idle"


def test_seek_backward_rewinds_cursor():
    """倒带：游标回退到更早时刻，事件同样只标目标时刻。"""
    client = TestClient(app)
    sid = make_session(client)
    seek(client, sid, 1500)

    body = seek(client, sid, 500).json()
    assert body["cursor_ms"] == 500
    # 1500 处可见 s2、s3；500 处仅 s1 可见
    assert _triples(body) == [
        (500, "exit", "s2"),
        (500, "exit", "s3"),
        (500, "enter", "s1"),
    ]
    assert [a["id"] for a in body["active"]] == ["s1"]


def test_seek_skipped_interval_events_are_not_replayed():
    """跳过区间不补发事件：跨过的进入/退出不出现在事件里。"""
    client = TestClient(app)
    sid = make_session(client)
    # 从 0 直接定位到 3000：跨过了 s2 的进入(800)、s1 的退出(1000)、
    # s3 的进入(1000)与退出(2500)、s2 的退出(3000)，一律不补发
    body = seek(client, sid, 3000).json()
    # 0 处仅 s1 可见 -> 一条 exit；3000 处无任何活动（s2 在 3000 退出，右开）
    assert body["events"] == [
        {"id": "s1", "speaker": "Alice", "track": 1,
         "time_ms": 3000, "type": "exit"},
    ]
    assert body["active"] == []
    assert all(e["time_ms"] == 3000 for e in body["events"])


def test_seek_same_position_reanchors_visible_active():
    """定位到当前时刻：两边都活动者同样退出再进入（显示整体重挂）。"""
    client = TestClient(app)
    sid = make_session(client)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    body = seek(client, sid, 900).json()
    assert _triples(body) == [
        (900, "exit", "s1"),
        (900, "exit", "s2"),
        (900, "enter", "s1"),
        (900, "enter", "s2"),
    ]
    assert [a["id"] for a in body["active"]] == ["s1", "s2"]


def test_seek_to_zero_and_past_the_end():
    """定位到 0 与越过全部字幕的末尾。"""
    client = TestClient(app)
    sid = make_session(client)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    # 回到 0：s1 在 0 处活动（左闭），s2 尚未开始
    body = seek(client, sid, 0).json()
    assert _triples(body) == [
        (0, "exit", "s1"),
        (0, "exit", "s2"),
        (0, "enter", "s1"),
    ]
    assert [a["id"] for a in body["active"]] == ["s1"]

    # 越过末尾：只有退出，没有进入
    body = seek(client, sid, 9999).json()
    assert _triples(body) == [(9999, "exit", "s1")]
    assert body["active"] == []


def test_seek_event_order_exits_first_then_enters_by_subtitle_order():
    """先 exit 后 enter；同类按会话字幕顺序（与字幕时间无关）。"""
    client = TestClient(app)
    resp = client.put("/scenes/scene-seek-order", json={
        "track_count": 3,
        "expected_revision": 0,
        "subtitles": [
            {"id": "a", "speaker": "A", "start_ms": 0, "end_ms": 100},
            {"id": "b", "speaker": "B", "start_ms": 0, "end_ms": 100},
            {"id": "c", "speaker": "C", "start_ms": 0, "end_ms": 100},
            {"id": "d", "speaker": "D", "start_ms": 500, "end_ms": 600},
            {"id": "e", "speaker": "E", "start_ms": 500, "end_ms": 600},
        ],
    })
    assert resp.status_code == 200
    sid = client.post(
        "/sessions", json={"scene_id": "scene-seek-order"}
    ).json()["session_id"]
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 50})

    body = seek(client, sid, 550).json()
    assert _triples(body) == [
        (550, "exit", "a"),
        (550, "exit", "b"),
        (550, "exit", "c"),
        (550, "enter", "d"),
        (550, "enter", "e"),
    ]


# ---------------------------------------------------------------------------
# 隐藏集合与快照
# ---------------------------------------------------------------------------


def test_seek_respects_hidden_speakers_on_both_sides():
    """隐藏说话人的字幕两端都不算可见：不发事件、不进 active。"""
    client = TestClient(app)
    sid = make_session(client)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})
    client.post(
        f"/sessions/{sid}/hidden_speakers",
        json={"hidden_speakers": ["Alice"]},
    )

    # 900 处可见仅 s2（s1 隐藏）；1500 处可见仅 s2（s3 是 Alice）
    body = seek(client, sid, 1500).json()
    assert _triples(body) == [(1500, "exit", "s2"), (1500, "enter", "s2")]
    assert [a["id"] for a in body["active"]] == ["s2"]

    # 隐藏集合不被定位改变：推进到 3000 后 Alice 的字幕仍不可见
    body = client.post(f"/sessions/{sid}/advance", json={"time_ms": 3000}).json()
    assert body["active"] == []


def test_seek_does_not_change_snapshot_or_hidden_set():
    """定位不改快照与隐藏集合：finish 的最终快照与场景一致。"""
    client = TestClient(app)
    sid = make_session(client)
    client.post(
        f"/sessions/{sid}/hidden_speakers",
        json={"hidden_speakers": ["Bob"]},
    )
    seek(client, sid, 1500)
    seek(client, sid, 100)

    body = client.post(f"/sessions/{sid}/finish").json()
    assert body["cursor_ms"] == 100
    assert [s["id"] for s in body["subtitles"]] == ["s1", "s2", "s3"]
    assert body["assignments"] == [
        {"id": "s1", "track": 1},
        {"id": "s2", "track": 2},
        {"id": "s3", "track": 1},
    ]
    assert body["changes"] == 0
    # 完整快照不含隐藏集合字段（隐藏只是播放期视图）
    assert "hidden_speakers" not in body


# ---------------------------------------------------------------------------
# 补取通道：队列与确认边界保留，新事件接续 seq
# ---------------------------------------------------------------------------


def test_seek_events_continue_channel_seq_and_boundaries_kept():
    """已开通道：定位事件接续 seq 入队，未确认队列与确认边界保留。"""
    client = TestClient(app)
    sid = make_session(client)
    open_channel(client, sid)
    # 推进到 1000：s2 enter@800、s1 exit@1000、s3 enter@1000 入队为 seq 1~3
    client.post(f"/sessions/{sid}/events", json={"time_ms": 1000})
    # 先补取建立“已返回”上界，再确认前 2 条：确认边界前进到 2
    body = fetch(client, sid, 0).json()
    assert [e["seq"] for e in body["events"]] == [1, 2, 3]
    body = fetch(client, sid, 2).json()
    assert body["acked_seq"] == 2
    assert [e["seq"] for e in body["events"]] == [3]

    # 倒带回 900：1000 处可见 s2、s3；900 处可见 s1、s2
    body = seek(client, sid, 900).json()
    assert _triples(body) == [
        (900, "exit", "s2"),
        (900, "exit", "s3"),
        (900, "enter", "s1"),
        (900, "enter", "s2"),
    ]

    # 补取：确认边界未动（仍 2），定位事件接续为 seq 4~7（时间回退仍按 seq）
    body = fetch(client, sid, 2).json()
    assert body["acked_seq"] == 2
    assert [
        (e["seq"], e["time_ms"], e["type"], e["id"]) for e in body["events"]
    ] == [
        (3, 1000, "enter", "s3"),
        (4, 900, "exit", "s2"),
        (5, 900, "exit", "s3"),
        (6, 900, "enter", "s1"),
        (7, 900, "enter", "s2"),
    ]
    # 越界校验仍以“已返回的最大 seq”为上界：ack 8 越界
    assert fetch(client, sid, 8).status_code == 422
    # 确认到 7 后队列清空
    body = fetch(client, sid, 7).json()
    assert body["acked_seq"] == 7
    assert body["events"] == []


def test_seek_without_channel_never_consumes_capacity():
    """未开通道的定位不产生容量占用，也不受容量约束。"""
    client = TestClient(app)
    sid = make_session(client)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})
    body = seek(client, sid, 1500).json()
    assert len(body["events"]) == 4
    # 没有通道：补取 404，定位本身不受任何容量影响
    assert fetch(client, sid, 0).status_code == 404


# ---------------------------------------------------------------------------
# 共享容量：不足 503，状态全不变；释放后重试成功
# ---------------------------------------------------------------------------


def make_client_with_limit(monkeypatch, limit):
    """配置 CHANNEL_EVENT_LIMIT 并重建惰性播放管理器，返回全新 TestClient。"""
    monkeypatch.setenv("CHANNEL_EVENT_LIMIT", str(limit))
    main_module.reset_playback()
    return TestClient(app)


def test_seek_capacity_exhausted_503_changes_nothing(monkeypatch):
    """满额 503：游标、队列、seq、确认边界与播放状态均不变；释放后可重试。"""
    client = make_client_with_limit(monkeypatch, 4)
    sid = make_session(client)
    open_channel(client, sid)
    # 3 条事件入队（seq 1~3），占用 3 个名额，仅剩 1
    client.post(f"/sessions/{sid}/events", json={"time_ms": 1000})

    # 定位到 900 会产生 4 条显示事件，超过剩余容量 1 -> 503
    resp = seek(client, sid, 900)
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["limit"] == 4
    assert detail["requested"] == 4
    assert detail["available"] == 1

    # 游标未动（仍 1000）：等时推进返回 1000 处的可见 active
    body = client.post(
        f"/sessions/{sid}/advance", json={"time_ms": 1000}
    ).json()
    assert body["cursor_ms"] == 1000
    assert [a["id"] for a in body["active"]] == ["s2", "s3"]
    # 队列与 seq 未变：仍只有 seq 1~3 未确认，确认边界仍 0
    body = fetch(client, sid, 0).json()
    assert body["acked_seq"] == 0
    assert [e["seq"] for e in body["events"]] == [1, 2, 3]
    # 播放状态未变（从未启动仍为 idle）
    assert autoplay_status(client, sid)["status"] == "idle"

    # 确认并删除前缀释放容量后，同一请求重试成功：事件接续 seq 4~7
    fetch(client, sid, 3)
    body = seek(client, sid, 900).json()
    assert body["cursor_ms"] == 900
    body = fetch(client, sid, 3).json()
    assert [e["seq"] for e in body["events"]] == [4, 5, 6, 7]


def test_seek_zero_events_succeeds_at_full_capacity(monkeypatch):
    """无显示事件的定位（两侧皆空）不占名额，满额时仍可执行。"""
    client = make_client_with_limit(monkeypatch, 3)
    # 占额会话：3 条事件占满全部共享容量
    hog = make_session(client, scene_id="scene-seek-hog")
    open_channel(client, hog)
    client.post(f"/sessions/{hog}/events", json={"time_ms": 1000})

    # 主会话：先推进到全部字幕结束之后（未开通道不占容量），再开通道
    sid = make_session(client)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 4000})
    open_channel(client, sid)
    # 4000 -> 5000：两侧皆无可见字幕，0 个事件，满额仍 200
    resp = seek(client, sid, 5000)
    assert resp.status_code == 200
    assert resp.json()["events"] == []
    assert resp.json()["active"] == []


# ---------------------------------------------------------------------------
# 自动播放：定位后除未启动外均为暂停；旧驱动不再推进；恢复重新计时
# ---------------------------------------------------------------------------


def test_seek_running_autoplay_becomes_paused_then_resume_retimes():
    """运行中定位 -> 暂停；定位返回后旧驱动不再推进；恢复从新游标重新计时。"""
    client = TestClient(app)
    sid = make_session(client)
    open_channel(client, sid)
    assert client.post(f"/sessions/{sid}/autoplay/start").status_code == 200
    # 等后台推进一段，确认确实在 running
    wait_for(lambda: autoplay_status(client, sid)["cursor_ms"] > 0)

    # 运行中定位到 5000：成功并转为暂停
    resp = seek(client, sid, 5000)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "paused"
    assert body["autoplay_status"] == "paused"
    assert body["cursor_ms"] == 5000

    # 定位返回后旧驱动不得推进：游标停在 5000
    time.sleep(0.3)
    state = autoplay_status(client, sid)
    assert state == {"session_id": sid, "status": "paused", "cursor_ms": 5000}

    # 恢复：从新游标 5000 重新计时，不追赶停顿
    resp = client.post(f"/sessions/{sid}/autoplay/resume")
    assert resp.status_code == 200
    assert resp.json()["cursor_ms"] == 5000
    wait_for(lambda: autoplay_status(client, sid)["cursor_ms"] > 5000)
    # 只计恢复后真实经过的时间：远小于“追赶”会到达的值
    assert autoplay_status(client, sid)["cursor_ms"] < 6000


def test_seek_paused_autoplay_stays_paused():
    """暂停中定位：仍为暂停，游标移到目标时刻。"""
    client = TestClient(app)
    sid = make_session(client)
    open_channel(client, sid)
    client.post(f"/sessions/{sid}/autoplay/start")
    wait_for(lambda: autoplay_status(client, sid)["cursor_ms"] > 0)
    client.post(f"/sessions/{sid}/autoplay/pause")

    body = seek(client, sid, 2500).json()
    assert body["status"] == "paused"
    assert body["cursor_ms"] == 2500
    time.sleep(0.2)
    assert autoplay_status(client, sid)["cursor_ms"] == 2500


def _record(scene_id, subtitles, track_count=2):
    subs = [dict(s) for s in subtitles]
    assignments = [
        {"id": s["id"], "track": i % track_count + 1}
        for i, s in enumerate(subs)
    ]
    return SceneRecord(
        scene_id=scene_id,
        revision=1,
        track_count=track_count,
        subtitles=subs,
        assignments=assignments,
        changes=0,
    )


def _open_session(pm, record):
    state = pm.create(record)
    session = pm.get(state["session_id"])
    pm.open_channel(session)
    return session


def test_seek_blocked_autoplay_becomes_paused_and_resume_retimes():
    """确定性回归：blocked 定位成功转为 paused；恢复从新游标重新计时。"""
    pm = PlaybackManager(channel_event_limit=10, autoplay_tick_interval_s=None)
    # 占额会话：9 条 enter 占去 9 个名额，剩余 1
    hog_subs = [
        {"id": f"h{i}", "speaker": f"H{i}",
         "start_ms": 10 + i * 10, "end_ms": 10 + i * 10 + 500}
        for i in range(9)
    ]
    hog = _open_session(pm, _record("scene-hog", hog_subs))
    pm.advance_events(hog, 90)
    assert pm.channel_capacity.used == 9

    # 主会话：b/c 在 100 同时进入（整批 2 条超过剩余 1 个名额 -> blocked）
    main = _open_session(pm, _record("scene-main", [
        {"id": "b", "speaker": "B", "start_ms": 100, "end_ms": 200},
        {"id": "c", "speaker": "C", "start_ms": 100, "end_ms": 200},
    ]))
    pm.start_autoplay(main, now=0.0)
    pm.tick_autoplay(now=0.050)   # 游标 50，无事件
    pm.tick_autoplay(now=0.100)   # 整批 2 条放不下 -> blocked，游标仍 50
    assert main.autoplay_status == AUTOPLAY_BLOCKED
    assert main.cursor_ms == 50

    # 阻塞中定位到 5000：两侧皆无可见字幕（0 个事件，不占名额）-> 成功
    state = pm.seek(main, 5000)
    assert state["status"] == "paused"
    assert state["cursor_ms"] == 5000
    assert state["events"] == []
    assert main.autoplay_status != AUTOPLAY_BLOCKED
    # 计时段不再推进：tick 不改变游标
    pm.tick_autoplay(now=30.0)
    assert main.cursor_ms == 5000

    # 恢复：从新游标 5000 重新计时，不追赶
    pm.resume_autoplay(main, now=40.0)
    pm.tick_autoplay(now=40.125)
    assert main.cursor_ms == 5125
    assert main.autoplay_status == AUTOPLAY_RUNNING


def test_seek_503_keeps_running_status_and_timing_segment():
    """确定性回归：运行中定位 503，状态、游标、队列、seq 与计时段均不变。"""
    pm = PlaybackManager(channel_event_limit=4, autoplay_tick_interval_s=None)
    main = _open_session(pm, _record("scene-main", [
        {"id": "a", "speaker": "A", "start_ms": 0, "end_ms": 60000},
        {"id": "b", "speaker": "B", "start_ms": 100, "end_ms": 200},
        {"id": "c", "speaker": "C", "start_ms": 100, "end_ms": 200},
    ]))
    pm.start_autoplay(main, now=0.0)
    pm.tick_autoplay(now=0.100)  # b/c enter 入队（used=2，剩余 2）
    assert main.autoplay_status == AUTOPLAY_RUNNING
    assert main.cursor_ms == 100
    assert main.channel.next_seq == 3

    # 定位到 5000 需要 4 条事件（a/b/c 退出 + a 进入），剩余容量 2 -> 503
    with pytest.raises(ChannelCapacityExhausted):
        pm.seek(main, 5000)
    # 播放状态、游标、seq、队列与计时段（base_*）全部不变
    assert main.autoplay_status == AUTOPLAY_RUNNING
    assert main.cursor_ms == 100
    assert main.base_cursor_ms == 0
    assert main.base_monotonic_s == 0.0
    assert main.channel.next_seq == 3
    assert [e["seq"] for e in main.channel.events] == [1, 2]
    assert pm.channel_capacity.used == 2

    # 旧驱动仍按原计时段推进：下一拍窗口 (100,150] 无事件，游标到 150
    pm.tick_autoplay(now=0.150)
    assert main.cursor_ms == 150

    # 确认释放容量后同一定位成功：转为暂停，事件接续 seq 3~6
    pm.fetch_channel(main, 0)   # 建立“已返回”上界
    pm.fetch_channel(main, 2)   # 确认并删除前缀，释放 2 个名额
    state = pm.seek(main, 5000)
    assert state["status"] == "paused"
    assert main.cursor_ms == 5000
    assert main.channel.next_seq == 7
    assert [e["seq"] for e in main.channel.events] == [3, 4, 5, 6]


# ---------------------------------------------------------------------------
# 404 / 422
# ---------------------------------------------------------------------------


def test_seek_unknown_session_404():
    client = TestClient(app)
    resp = seek(client, "no-such-session", 1000)
    assert resp.status_code == 404


def test_seek_finished_session_404():
    client = TestClient(app)
    sid = make_session(client)
    assert client.post(f"/sessions/{sid}/finish").status_code == 200
    assert seek(client, sid, 1000).status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"time_ms": -1},        # 负数
        {"time_ms": True},      # 布尔不隐式转换
        {"time_ms": False},
        {"time_ms": "1000"},    # 数字字符串不隐式转换
        {"time_ms": 1.5},       # 浮点
        {"time_ms": None},      # null
        {},                     # 字段缺失
    ],
)
def test_seek_invalid_time_422_and_state_unchanged(payload):
    client = TestClient(app)
    sid = make_session(client)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    resp = client.post(f"/sessions/{sid}/seek", json=payload)
    assert resp.status_code == 422
    # 会话不变：游标仍 900
    body = client.post(f"/sessions/{sid}/advance", json={"time_ms": 900}).json()
    assert body["cursor_ms"] == 900
    assert [a["id"] for a in body["active"]] == ["s1", "s2"]


# ---------------------------------------------------------------------------
# 串行化与旧接口
# ---------------------------------------------------------------------------


def test_concurrent_seeks_serialized():
    """并发定位在会话锁内串行：全部 200，各响应自洽无中间态。"""
    client = TestClient(app)
    sid = make_session(client)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    results = []
    barrier = threading.Barrier(5)

    def worker(target):
        local = TestClient(app)
        barrier.wait()
        resp = seek(local, sid, target)
        results.append(
            (resp.status_code, resp.json() if resp.status_code == 200 else None)
        )

    targets = [1500, 500, 3000, 0, 2000]
    threads = [threading.Thread(target=worker, args=(t,)) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(status == 200 for status, _ in results)
    for _, body in results:
        # 每个定位响应自洽：事件均标目标时刻、先 exit 后 enter、
        # active 与游标一致
        assert all(e["time_ms"] == body["cursor_ms"] for e in body["events"])
        types = [e["type"] for e in body["events"]]
        assert types == sorted(types, key=lambda t: t != "exit")
        for a in body["active"]:
            assert a["start_ms"] <= body["cursor_ms"] < a["end_ms"]
    # 最终游标必为某次定位的目标（最后一次生效的定位）
    body = client.post(f"/sessions/{sid}/finish").json()
    assert body["cursor_ms"] in set(targets)


def test_seek_serialized_with_finish():
    """定位与结束串行：结束后才轮到生效的定位返回 404。"""
    client = TestClient(app)
    sid = make_session(client)
    assert client.post(f"/sessions/{sid}/finish").status_code == 200
    assert seek(client, sid, 1000).status_code == 404


def test_advance_still_forbids_regression_after_seek_back():
    """定位回退后，原推进接口仍禁止倒退（只能不减）。"""
    client = TestClient(app)
    sid = make_session(client)
    seek(client, sid, 1500)
    seek(client, sid, 500)  # 倒带回 500

    # 推进到 400 仍倒退 -> 422；推进到 600 正常
    resp = client.post(f"/sessions/{sid}/advance", json={"time_ms": 400})
    assert resp.status_code == 422
    body = client.post(f"/sessions/{sid}/advance", json={"time_ms": 600}).json()
    assert body["cursor_ms"] == 600
    assert [a["id"] for a in body["active"]] == ["s1"]


def test_old_interfaces_response_fields_unchanged():
    """旧接口响应不含定位新增字段（status/autoplay_status）。"""
    client = TestClient(app)
    sid = make_session(client)

    body = client.post(f"/sessions/{sid}/advance", json={"time_ms": 900}).json()
    assert set(body) == {
        "session_id", "scene_id", "source_revision", "cursor_ms", "active",
    }
    body = client.post(f"/sessions/{sid}/events", json={"time_ms": 1000}).json()
    assert set(body) == {
        "session_id", "scene_id", "source_revision", "cursor_ms",
        "events", "active",
    }
    body = client.post(
        f"/sessions/{sid}/hidden_speakers",
        json={"hidden_speakers": ["Alice"]},
    ).json()
    assert set(body) == {
        "session_id", "scene_id", "source_revision", "cursor_ms",
        "hidden_speakers", "events", "active",
    }
